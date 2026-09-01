import pandas as pd
import shutil
import os
from pathlib import Path

def extract_audio_files(csv_file, audio_source_dir, output_dir):
    """
    从CSV文件中提取音频文件并复制到指定目录
    
    参数:
    csv_file: CSV文件路径
    audio_source_dir: 音频文件源目录
    output_dir: 输出目录
    """
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file, sep='\t')  # 使用制表符分隔
        print(f"成功读取CSV文件，共{len(df)}行数据")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 统计信息
    successful_copies = 0
    failed_copies = 0
    
    # 遍历每一行，复制音频文件
    for index, row in df.iterrows():
        filename = row['filename']
        
        # 构建源文件路径和目标文件路径
        source_path = os.path.join(audio_source_dir, filename)
        target_path = os.path.join(output_dir, os.path.basename(filename))
        
        try:
            # 复制文件
            shutil.copy2(source_path, target_path)
            successful_copies += 1
            if successful_copies % 100 == 0:  # 每100个文件打印一次进度
                print(f"已复制 {successful_copies} 个文件...")
                
        except FileNotFoundError:
            print(f"文件未找到: {source_path}")
            failed_copies += 1
        except Exception as e:
            print(f"复制文件 {filename} 时出错: {e}")
            failed_copies += 1
    
    # 打印统计信息
    print("\n=== 复制完成 ===")
    print(f"成功复制: {successful_copies} 个文件")
    print(f"失败: {failed_copies} 个文件")
    print(f"输出目录: {output_dir}")

def main():
    # 配置参数
    csv_file = "split25.csv"  # CSV文件路径
    audio_source_dir = "audio"  # 音频文件源目录
    output_dir = "extracted_audio"  # 输出目录
    
    # 检查源目录是否存在
    if not os.path.exists(audio_source_dir):
        print(f"错误: 音频源目录 '{audio_source_dir}' 不存在")
        return
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: CSV文件 '{csv_file}' 不存在")
        return
    
    # 执行提取
    extract_audio_files(csv_file, audio_source_dir, output_dir)

if __name__ == "__main__":
    main()