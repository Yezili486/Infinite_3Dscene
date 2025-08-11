// 演示流程控制
document.getElementById('startDemo').addEventListener('click', function() {
    // 显示处理过程
    document.getElementById('processing').classList.remove('hidden');
    
    // 平滑滚动到处理过程
    document.getElementById('processing').scrollIntoView({ behavior: 'smooth' });
    
    // 模拟进度
    simulateProgress();
});

// 代码展示控制
document.getElementById('showCode').addEventListener('click', function() {
    document.getElementById('code').classList.remove('hidden');
    document.getElementById('code').scrollIntoView({ behavior: 'smooth' });
});

// 代码标签页切换
document.querySelectorAll('.tab-btn').forEach(button => {
    button.addEventListener('click', function() {
        // 移除所有标签的活跃状态
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // 添加当前标签的活跃状态
        this.classList.add('active');
        
        // 隐藏所有代码内容
        document.querySelectorAll('.code-block').forEach(content => {
            content.classList.remove('active');
        });
        
        // 显示对应代码内容
        const target = this.getAttribute('data-target');
        document.getElementById(target).classList.add('active');
    });
});

// 模拟处理进度
function simulateProgress() {
    // 模型加载
    let modelProgress = 0;
    document.getElementById('modelStatus').className = 'status processing';
    document.getElementById('modelStatus').textContent = '加载中';
    
    const modelInterval = setInterval(() => {
        modelProgress += 1;
        document.getElementById('modelProgress').style.width = modelProgress + '%';
        if (modelProgress >= 100) {
            clearInterval(modelInterval);
            document.getElementById('modelStatus').className = 'status completed';
            document.getElementById('modelStatus').textContent = '完成';
            
            // 开始生成图像
            setTimeout(() => {
                let imageProgress = 0;
                document.getElementById('imageStatus').className = 'status processing';
                document.getElementById('imageStatus').textContent = '生成中';
                
                const imageInterval = setInterval(() => {
                    imageProgress += 2;
                    document.getElementById('imageProgress').style.width = imageProgress + '%';
                    if (imageProgress >= 100) {
                        clearInterval(imageInterval);
                        document.getElementById('imageStatus').className = 'status completed';
                        document.getElementById('imageStatus').textContent = '完成';
                        
                        // 开始3D重建
                        setTimeout(() => {
                            let reconstructionProgress = 0;
                            document.getElementById('reconstructionStatus').className = 'status processing';
                            document.getElementById('reconstructionStatus').textContent = '重建中';
                            
                            const reconstructionInterval = setInterval(() => {
                                reconstructionProgress += 1;
                                document.getElementById('reconstructionProgress').style.width = reconstructionProgress + '%';
                                if (reconstructionProgress >= 100) {
                                    clearInterval(reconstructionInterval);
                                    document.getElementById('reconstructionStatus').className = 'status completed';
                                    document.getElementById('reconstructionStatus').textContent = '完成';
                                    
                                    // 开始相机对齐
                                    setTimeout(() => {
                                        let alignmentProgress = 0;
                                        document.getElementById('alignmentStatus').className = 'status processing';
                                        document.getElementById('alignmentStatus').textContent = '对齐中';
                                        
                                        const alignmentInterval = setInterval(() => {
                                            alignmentProgress += 2;
                                            document.getElementById('alignmentProgress').style.width = alignmentProgress + '%';
                                            if (alignmentProgress >= 100) {
                                                clearInterval(alignmentInterval);
                                                document.getElementById('alignmentStatus').className = 'status completed';
                                                document.getElementById('alignmentStatus').textContent = '完成';
                                                
                                                // 显示结果
                                                setTimeout(() => {
                                                    document.getElementById('results').classList.remove('hidden');
                                                    document.getElementById('parameters').classList.remove('hidden');
                                                    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
                                                    
                                                    // 绘制误差图表
                                                    drawErrorChart();
                                                }, 1000);
                                            }
                                        }, 100);
                                    }, 1000);
                                }
                            }, 80);
                        }, 1000);
                    }
                }, 50);
            }, 1000);
        }
    }, 50);
}

// 绘制误差变化图表
function drawErrorChart() {
    const ctx = document.getElementById('errorChart').getContext('2d');
    
    // 模拟相机对齐误差数据
    const data = {
        labels: Array.from({length: 20}, (_, i) => `Step ${i+1}`),
        datasets: [{
            label: '视角对齐误差',
            data: [0.15, 0.13, 0.11, 0.095, 0.082, 0.075, 0.068, 0.062, 0.057, 0.052, 
                   0.048, 0.043, 0.039, 0.035, 0.031, 0.028, 0.026, 0.024, 0.023, 0.023],
            borderColor: '#165DFF',
            backgroundColor: 'rgba(22, 93, 255, 0.1)',
            fill: true,
            tension: 0.4
        }]
    };
    
    new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '误差值'
                    }
                }
            }
        }
    });
}

// 相机视角动画控制 - 同一物体连续视角变化
let animationInterval;
let currentImageIndex = 0;
// 同一闹钟的不同角度图片序列
const animationImages = [
    '../assets/images/alarm-clock-0.jpg',  // 0° 正面
    '../assets/images/alarm-clock-30.jpg', // 30° 斜侧
    '../assets/images/alarm-clock-45.jpg', // 45° 斜侧
    '../assets/images/alarm-clock-60.jpg', // 60° 斜侧
    '../assets/images/alarm-clock-90.jpg'  // 90° 侧面
];

document.getElementById('playAnimation').addEventListener('click', function() {
    if (animationInterval) clearInterval(animationInterval);
    
    animationInterval = setInterval(() => {
        currentImageIndex = (currentImageIndex + 1) % animationImages.length;
        document.getElementById('cameraAnimation').src = animationImages[currentImageIndex];
    }, 1000);
});

document.getElementById('pauseAnimation').addEventListener('click', function() {
    clearInterval(animationInterval);
});

// 平滑滚动所有锚点链接
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        document.querySelector(targetId).scrollIntoView({
            behavior: 'smooth'
        });
    });
});
