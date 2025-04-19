import cv2
import numpy as np

def get_dark_channel(image, window_size):
    """计算图像的暗通道。"""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def get_atmospheric_light(image, dark_channel, percentile=0.001):
    """估算大气光。"""
    image_h, image_w = image.shape[:2]
    dark_h, dark_w = dark_channel.shape
    num_pixels = dark_h * dark_w
    num_brightest = int(num_pixels * percentile)

    # 展平暗通道并获取排序后像素的索引
    flat_dark_channel = dark_channel.ravel()
    indices = np.argsort(flat_dark_channel)[::-1] # 降序排序

    # 选择最亮的顶部像素
    brightest_indices = indices[:num_brightest]

    # 将扁平化索引转换为二维坐标
    row_indices, col_indices = np.unravel_index(brightest_indices, (dark_h, dark_w))

    atmospheric_light = np.zeros(3)
    max_intensity = 0

    # 在暗通道中最亮的像素中，找到原始图像中强度最高的像素
    for i in range(num_brightest):
        row, col = row_indices[i], col_indices[i]
        # 使用原始图像中对应于暗通道中前百分位像素的最亮像素作为大气光估计值。
        if np.sum(image[row, col]) > max_intensity:
            max_intensity = np.sum(image[row, col])
            atmospheric_light = image[row, col]

    return atmospheric_light

def get_transmission(image, atmospheric_light, window_size, omega=0.95):
    """估算透射率图。"""
    # 如果大气光通道为零，避免除零错误
    atmospheric_light_safe = np.maximum(atmospheric_light, 1e-6)
    normalized_image = image / atmospheric_light_safe
    dark_channel_normalized = get_dark_channel(normalized_image, window_size)
    transmission = 1 - omega * dark_channel_normalized
    return transmission

def dehaze(image, atmospheric_light, transmission, t0=0.1):
    """恢复无雾图像。"""
    # 限制透射率以避免除以零或接近零
    transmission_clamped = np.maximum(transmission, t0)

    # 确保透射率有3个通道以进行广播
    transmission_3d = np.expand_dims(transmission_clamped, axis=2)
    transmission_3d = np.repeat(transmission_3d, 3, axis=2)

    # 恢复场景亮度（去雾后的图像）
    # J = (I - A) / t + A
    dehazed_image = (image - atmospheric_light) / transmission_3d + atmospheric_light

    # 将值裁剪到 [0, 1] 范围
    dehazed_image = np.clip(dehazed_image, 0, 1)
    return dehazed_image

if __name__ == "__main__":
    # --- 参数 --- 用户请求的输入图像
    input_image_path = '0001_gt.png'
    output_image_path = 'dehazed_image.png'
    window_size = 15 # 暗通道和透射率估计的块大小
    omega = 0.98     # 透射率估计因子 (原为 0.95)
    t0 = 0.1         # 透射率下界
    percentile = 0.001 # 用于大气光估计的最亮像素百分比
    # ------------------

    # 加载图像
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"错误：无法从 {input_image_path} 加载图像")
        # 检查文件是否存在
        import os
        if not os.path.exists(input_image_path):
            print(f"文件未找到: {os.path.abspath(input_image_path)}")
        exit()

    # 将图像归一化到 [0, 1] 范围并转换为浮点数
    img_float = img.astype(np.float64) / 255.0

    # 计算暗通道
    dark_channel = get_dark_channel(img_float, window_size)

    # 估算大气光
    atmospheric_light = get_atmospheric_light(img_float, dark_channel, percentile)
    print(f"估算的大气光 (A): {atmospheric_light * 255}")

    # 估算透射率图
    transmission = get_transmission(img_float, atmospheric_light, window_size, omega)

    # 对图像进行去雾
    dehazed_img_float = dehaze(img_float, atmospheric_light, transmission, t0)

    # 转换回 [0, 255] 范围和 uint8 类型
    dehazed_img_uint8 = (dehazed_img_float * 255).astype(np.uint8)

    # 保存结果
    cv2.imwrite(output_image_path, dehazed_img_uint8)
    print(f"去雾后的图像已保存至 {output_image_path}")
