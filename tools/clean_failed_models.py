import os
import shutil
import argparse
from datetime import datetime

def get_save_dir():
    """获取save目录的绝对路径"""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(current_dir, 'save')

def list_model_dirs():
    """列出所有模型目录"""
    save_dir = get_save_dir()
    if not os.path.exists(save_dir):
        print(f"Save目录不存在: {save_dir}")
        return []
    
    dirs = []
    for item in os.listdir(save_dir):
        item_path = os.path.join(save_dir, item)
        if os.path.isdir(item_path):
            dirs.append(item)
    return dirs

def delete_model_dir(dir_name):
    """删除指定的模型目录"""
    save_dir = get_save_dir()
    dir_path = os.path.join(save_dir, dir_name)
    
    if not os.path.exists(dir_path):
        print(f"错误：目录不存在: {dir_name}")
        return False
    
    try:
        shutil.rmtree(dir_path)
        print(f"成功删除目录: {dir_name}")
        return True
    except Exception as e:
        print(f"删除目录失败 {dir_name}: {str(e)}")
        return False

def interactive_mode():
    """交互式模式"""
    print("\n=== 失败模型清理工具 ===")
    print("当前时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 列出所有模型目录
    model_dirs = list_model_dirs()
    if not model_dirs:
        print("\n没有找到任何模型目录！")
        return
    
    print("\n找到以下模型目录:")
    for i, dir_name in enumerate(model_dirs, 1):
        print(f"{i}. {dir_name}")
    
    while True:
        print("\n请选择操作:")
        print("1. 删除指定目录")
        print("2. 退出程序")
        
        choice = input("\n请输入选项 (1-2): ").strip()
        
        if choice == "2":
            break
        elif choice == "1":
            print("\n请输入要删除的目录编号 (用逗号分隔多个编号):")
            indexes = input().strip()
            
            try:
                # 解析用户输入的编号
                index_list = [int(x.strip()) for x in indexes.split(",")]
                
                # 验证编号是否有效
                if any(i < 1 or i > len(model_dirs) for i in index_list):
                    print("错误：无效的目录编号！")
                    continue
                
                # 确认删除
                to_delete = [model_dirs[i-1] for i in index_list]
                print("\n将要删除以下目录:")
                for dir_name in to_delete:
                    print(f"- {dir_name}")
                
                confirm = input("\n确认删除这些目录? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("操作已取消")
                    continue
                
                # 执行删除
                success_count = 0
                for dir_name in to_delete:
                    if delete_model_dir(dir_name):
                        success_count += 1
                
                print(f"\n操作完成: 成功删除 {success_count}/{len(to_delete)} 个目录")
                
            except ValueError:
                print("错误：请输入有效的数字！")
            except Exception as e:
                print(f"发生错误: {str(e)}")
        else:
            print("无效的选项，请重试！")

def main():
    parser = argparse.ArgumentParser(description='模型目录清理工具')
    parser.add_argument('--dir', type=str, help='要删除的目录名称')
    args = parser.parse_args()

    if args.dir:
        # 命令行模式：直接删除指定目录
        delete_model_dir(args.dir)
    else:
        # 交互式模式
        interactive_mode()

if __name__ == "__main__":
    main() 