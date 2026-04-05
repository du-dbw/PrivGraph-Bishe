# 实验作业4
# 题目：复利的力量
# 功能：计算100美元在年利率20%下，30年后的金额（复利计算）

def main():
    # 本金（美元）
    principal = 100
    
    # 年利率（20%）
    rate = 0.2
    
    # 年数
    years = 30
    
    # 使用复利公式计算
    # 最终金额 = 本金 × (1 + 年利率) 的 年数次方
    amount = principal * (1 + rate) ** years
    
    # 输出结果（保留两位小数）
    print(f"30年后这笔钱变成 {amount:.2f} 美元")


# 程序入口
if __name__ == "__main__":
    main()