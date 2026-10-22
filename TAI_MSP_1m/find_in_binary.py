import struct

def parse_binary_file(filepath, frame_size=44, skip_bytes=24, frames_per_group=32,
                      float_offset=0, float_format='<f'):
    """
    读取二进制文件，按帧解析并分组，提取浮点数。

    参数:
        filepath: 二进制文件路径
        frame_size: 每帧字节数，默认44
        skip_bytes: 每帧跳过的前导字节数，默认24（取剩余部分）
        frames_per_group: 每组合并的帧数，默认32
        float_offset: 在每组bytes中，浮点数开始的字节偏移，默认0
        float_format: struct格式字符串，默认'<f'（小端单精度）
                     可改为'<d'（小端双精度）、'>f'（大端单精度）等

    返回:
        groups: 合并后的bytes列表
        floats: 从每组中提取的浮点数列表
    """
    # 1. 读取整个文件
    with open(filepath, 'rb') as f:
        data = f.read(12800000)

    # 2. 检查文件长度是否为帧大小的整数倍（非强制，仅提示）
    total_frames = len(data) // frame_size
    if len(data) % frame_size != 0:
        print(f"警告: 文件大小 {len(data)} 字节，不是 {frame_size} 的整数倍，剩余字节将忽略")

    # 3. 提取每帧的有效部分（skip_bytes之后）
    chunk_size = frame_size - skip_bytes   # 每帧有效数据字节数，44-24=20
    frames = []
    gps_timelist = []
    for i in range(100000):
        start = i * frame_size
        end = start + frame_size
        aux_frame = data[start: end]
        mili_seconds = int.from_bytes(aux_frame[14:16], 'big')
        micro_seconds = int.from_bytes(aux_frame[16:18], 'big')
        gps_seconds = int.from_bytes(aux_frame[18:22], 'big')
        gps_timelist.append(gps_seconds+0.001*mili_seconds+0.000001*micro_seconds)
        # frames.append(data[skip_bytes:])

    # 4. 每 frames_per_group 帧合并为一组
    groups = []
    for i in range(0, len(frames), frames_per_group):
        group_frames = frames[i:i + frames_per_group]
        if len(group_frames) < frames_per_group:
            print(f"最后一组帧数不足 {frames_per_group}，已丢弃 {len(group_frames)} 帧")
            break
        # 将所有帧的bytes拼接
        group_bytes = b''.join(group_frames)
        groups.append(group_bytes)

    # 5. 遍历每组，读取指定偏移处的浮点数
    float_size = struct.calcsize(float_format)  # 浮点数占用字节数
    floats = []
    for idx, group in enumerate(groups):
        if float_offset + float_size > len(group):
            print(f"组 {idx} 长度 {len(group)}，无法从偏移 {float_offset} 读取 {float_size} 字节浮点数，跳过")
            continue
        raw = group[float_offset:float_offset + float_size]
        value = struct.unpack(float_format, raw)[0]
        floats.append(value)

    return groups, floats

# 使用示例
if __name__ == '__main__':
    file_path = r"C:\Users\63441\Desktop\AS07_PMS_AUX.bin"
    groups, float_values = parse_binary_file(
        filepath=file_path,
        frame_size=44,
        skip_bytes=24,
        frames_per_group=32,
        float_offset=0,         # 从组内第0字节开始读
        float_format='<f'       # 小端单精度，若为大端双精度用 '>d'
    )
    print(f"共生成 {len(groups)} 个组，每组 {len(groups[0]) if groups else 0} 字节")
    print("提取的浮点数:", float_values)