"""
COLMAP data parser for LERF datasets
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import struct
from collections import namedtuple
import cv2

from .camera import Camera


# COLMAP data structures
CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera_COLMAP = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image_COLMAP = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

# Camera models supported by COLMAP
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path: Path) -> Dict[int, Camera_COLMAP]:
    """Read cameras.txt file"""
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera_COLMAP(
                    id=camera_id, model=model, width=width, height=height, params=params
                )
    return cameras


def read_cameras_binary(path: Path) -> Dict[int, Camera_COLMAP]:
    """Read cameras.bin file"""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params, format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera_COLMAP(
                id=camera_id, model=model_name, width=width, height=height, params=np.array(params)
            )
    return cameras


def read_images_text(path: Path) -> Dict[int, Image_COLMAP]:
    """Read images.txt file"""
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image_COLMAP(
                    id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids
                )
    return images


def read_images_binary(path: Path) -> Dict[int, Image_COLMAP]:
    """Read images.bin file"""
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D, format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(x_y_id_s[0::3]), tuple(x_y_id_s[1::3])])
            point3D_ids = np.array(tuple(x_y_id_s[2::3]))
            images[image_id] = Image_COLMAP(
                id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids
            )
    return images


def read_points3D_text(path: Path) -> Dict[int, Point3D]:
    """Read points3D.txt file"""
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id, xyz=xyz, rgb=rgb, error=error,
                    image_ids=image_ids, point2D_idxs=point2D_idxs
                )
    return points3D


def read_points3D_binary(path: Path) -> Dict[int, Point3D]:
    """Read points3D.bin file"""
    points3D = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8*track_length, format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(track_elems[0::2]))
            point2D_idxs = np.array(tuple(track_elems[1::2]))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb, error=error,
                image_ids=image_ids, point2D_idxs=point2D_idxs
            )
    return points3D


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def parse_colmap_reconstruction(colmap_path: Path, target_resolution: Tuple[int, int] = (512, 512)) -> Tuple[List[Camera], List[str]]:
    """
    Parse COLMAP reconstruction and convert to our Camera format
    
    Args:
        colmap_path: Path to COLMAP reconstruction directory
        target_resolution: Target image resolution
        
    Returns:
        List of Camera objects and corresponding image names
    """
    colmap_path = Path(colmap_path)
    
    # Read COLMAP data (try binary first, then text)
    cameras_file = colmap_path / "cameras.bin"
    images_file = colmap_path / "images.bin"
    points3D_file = colmap_path / "points3D.bin"
    
    if not cameras_file.exists():
        cameras_file = colmap_path / "cameras.txt"
        images_file = colmap_path / "images.txt" 
        points3D_file = colmap_path / "points3D.txt"
    
    if not cameras_file.exists():
        raise FileNotFoundError(f"No COLMAP cameras file found in {colmap_path}")
    
    # Read cameras
    if cameras_file.suffix == '.bin':
        colmap_cameras = read_cameras_binary(cameras_file)
    else:
        colmap_cameras = read_cameras_text(cameras_file)
    
    # Read images
    if images_file.exists():
        if images_file.suffix == '.bin':
            colmap_images = read_images_binary(images_file)
        else:
            colmap_images = read_images_text(images_file)
    else:
        colmap_images = {}
    
    # Convert to our Camera format
    cameras = []
    image_names = []
    
    for image_id, image in colmap_images.items():
        # Get camera parameters
        camera_id = image.camera_id
        colmap_camera = colmap_cameras[camera_id]
        
        # Extract camera intrinsics
        if colmap_camera.model == "PINHOLE":
            fx, fy, cx, cy = colmap_camera.params
        elif colmap_camera.model == "SIMPLE_PINHOLE":
            fx = fy = colmap_camera.params[0]
            cx, cy = colmap_camera.params[1], colmap_camera.params[2]
        else:
            # For other models, use the first 4 parameters as approximation
            params = colmap_camera.params
            fx = params[0] if len(params) > 0 else 800.0
            fy = params[1] if len(params) > 1 else fx
            cx = params[2] if len(params) > 2 else colmap_camera.width / 2
            cy = params[3] if len(params) > 3 else colmap_camera.height / 2
        
        # Scale intrinsics if resolution changes
        scale_x = target_resolution[0] / colmap_camera.width
        scale_y = target_resolution[1] / colmap_camera.height
        
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y
        
        # Convert COLMAP pose to camera-to-world matrix
        R = qvec2rotmat(image.qvec)
        t = image.tvec
        
        # COLMAP uses world-to-camera, we need camera-to-world
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = R.T
        camera_to_world[:3, 3] = -R.T @ t
        
        # Create Camera object
        camera = Camera(
            image_width=target_resolution[0],
            image_height=target_resolution[1],
            fx=fx, fy=fy, cx=cx, cy=cy,
            camera_to_world=torch.from_numpy(camera_to_world).float()
        )
        
        cameras.append(camera)
        image_names.append(image.name)
    
    return cameras, image_names


class COLMAPLoader:
    """COLMAP data loader for LERF datasets"""
    
    def __init__(self, colmap_path: Path, images_path: Optional[Path] = None):
        """
        Initialize COLMAP loader
        
        Args:
            colmap_path: Path to COLMAP reconstruction
            images_path: Path to images directory (if different from colmap_path)
        """
        self.colmap_path = Path(colmap_path)
        self.images_path = Path(images_path) if images_path else self.colmap_path / "images"
        
        # Load COLMAP data
        self.cameras, self.image_names = parse_colmap_reconstruction(self.colmap_path)
        
        print(f"Loaded COLMAP reconstruction with {len(self.cameras)} cameras")
    
    def get_cameras_and_images(self, target_resolution: Tuple[int, int] = (512, 512)) -> Tuple[List[Camera], List[torch.Tensor]]:
        """
        Get cameras and loaded images
        
        Args:
            target_resolution: Target image resolution
            
        Returns:
            List of cameras and corresponding images
        """
        from PIL import Image as PILImage
        
        images = []
        valid_cameras = []
        
        for camera, image_name in zip(self.cameras, self.image_names):
            image_path = self.images_path / image_name
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Load and preprocess image
            try:
                image = PILImage.open(image_path).convert('RGB')
                image = np.array(image) / 255.0
                
                # Resize to target resolution
                image = cv2.resize(image, target_resolution, interpolation=cv2.INTER_AREA)
                
                # Convert to tensor
                image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
                
                images.append(image_tensor)
                valid_cameras.append(camera)
                
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
        
        return valid_cameras, images


if __name__ == '__main__':
    # Test COLMAP parser
    print("Testing COLMAP parser...")
    
    # This would test with actual COLMAP data
    # colmap_loader = COLMAPLoader('/path/to/colmap/reconstruction')
    # cameras, images = colmap_loader.get_cameras_and_images()
    
    print("COLMAP parser ready for use with LERF datasets!")
