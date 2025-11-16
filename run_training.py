import subprocess
import sys
from datetime import datetime

# 定义所有训练命令
commands = [
    # XD 数据集任务
    {
        "name": "xd event",
        "dataset": "xd",
        "cmd": ["python", "train.py", "--dataset", "xd", "--batch_size", "128", "--clients_num", "6"]
    },
    {
        "name": "xd random",
        "dataset": "xd",
        "cmd": ["python", "train.py", "--dataset", "xd", "--batch_size", "128", "--clients_num", "10", "--split_mode", "random"]
    },
    {
        "name": "xd scene",
        "dataset": "xd",
        "cmd": ["python", "train.py", "--dataset", "xd", "--batch_size", "128", "--clients_num", "13", "--split_mode", "scene"]
    },
    # UCF 数据集任务
    {
        "name": "ucf event",
        "dataset": "ucf",
        "cmd": ["python", "train.py"]
    },
    {
        "name": "ucf random",
        "dataset": "ucf",
        "cmd": ["python", "train.py", "--split_mode", "random", "--clients_num", "10"]
    },
    {
        "name": "ucf scene",
        "dataset": "ucf",
        "cmd": ["python", "train.py", "--split_mode", "scene", "--clients_num", "9"]
    }
]


def run_command(name, cmd):
    """运行单个命令并等待其完成"""
    print(f"\n{'='*60}")
    print(f"开始执行: {name}")
    print(f"命令: {' '.join(cmd)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )

        print(f"\n{'='*60}")
        print(f"✓ 成功完成: {name}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"✗ 执行失败: {name}")
        print(f"错误码: {e.returncode}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"自动继续执行下一个任务...")
        print(f"{'='*60}\n")
        return False

    except KeyboardInterrupt:
        print(f"\n\n用户中断执行: {name}")
        raise


def main():
    """主函数：根据参数选择执行的数据集"""
    if len(sys.argv) < 2:
        print("❌ 请指定要运行的数据集，例如：")
        print("   python run_all.py xd")
        print("   python run_all.py ucf")
        print("   python run_all.py all")
        sys.exit(1)

    dataset_choice = sys.argv[1].lower()
    valid_choices = ["xd", "ucf", "all"]

    if dataset_choice not in valid_choices:
        print(f"❌ 无效数据集: {dataset_choice}")
        print("有效选项: xd / ucf / all")
        sys.exit(1)

    # 选择任务
    if dataset_choice == "all":
        selected_tasks = commands
    else:
        selected_tasks = [task for task in commands if task["dataset"] == dataset_choice]

    print(f"开始执行 {dataset_choice.upper()} 数据集任务...")
    print(f"共 {len(selected_tasks)} 个任务\n")

    success_count = 0
    failed_tasks = []

    try:
        current_dataset = None
        for i, task in enumerate(selected_tasks, 1):
            # 打印数据集分隔提示
            if task["dataset"] != current_dataset:
                current_dataset = task["dataset"]
                print(f"\n{'#'*60}")
                print(f"### 当前数据集: {current_dataset.upper()}")
                print(f"{'#'*60}\n")

            print(f"任务进度: {i}/{len(selected_tasks)}")
            success = run_command(task["name"], task["cmd"])

            if success:
                success_count += 1
            else:
                failed_tasks.append(task["name"])

    except KeyboardInterrupt:
        print("\n\n用户手动中断所有任务")

    # 输出总结
    print("\n" + "=" * 60)
    print("所有任务执行完毕")
    print(f"成功: {success_count}/{len(selected_tasks)}")
    if failed_tasks:
        print(f"失败的任务: {', '.join(failed_tasks)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
