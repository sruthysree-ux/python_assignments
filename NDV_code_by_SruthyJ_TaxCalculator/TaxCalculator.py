def calculate_tax(income):
    if income <= 250000:
        tax = 0
    elif income <= 500000:
        tax = (income - 250000) * 0.05
    elif income <= 1000000:
        tax = (250000 * 0.05) + (income - 500000) * 0.2
    else:
        tax = (250000 * 0.05) + (500000 * 0.2) + (income - 1000000) * 0.3
    return tax

try:
    income = float(input("Enter your annual income: "))
    tax = calculate_tax(income)
    total_income = income - tax
    print(f"Your payable tax is {tax}")
    print(f"your final income after deducting tax is {income} - {tax} = {total_income}")
except ValueError:
    print("Please enter a valid number for income.")
