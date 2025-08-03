class DemoConfig:
    def __init__(self, device="cuda"):
        self.device = device
        self.point_cloud_density = 300000
        self.render_iterations = 500
        self.detail_strength = 1.2
        self.render_views = [0, 45, 90]
        self.pretrained = {
            "esrgan": "./pretrained/esrgan.pth",
            "zoedepth": "./pretrained/zoedepth.pt",
            "closeup_gs": "./pretrained/closeup_gs.pt"
        } 