import pandas as pd
from lightweight_charts import Chart
import os
from tabulate import tabulate


def show_chart(symbol: str):
    df = pd.read_csv(f'../datasets/stock_data/{symbol}.csv')
    print(symbol, "Chart")
    chart = Chart()
    chart.set(df)
    chart.show(block=True)
    

def get_list_of_stocks(dataset_path):
    stock_list = []
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.csv'):
            stock_name = file_name.split('.')[0]
            stock_list.append(stock_name)
    return stock_list

def prompt_user_for_stock(dataset_path):
    stock_list = get_list_of_stocks(dataset_path)
    # Determine the number of rows and columns needed
    num_rows = (len(stock_list) + 4) // 5
    num_cols = min(len(stock_list), 5)
    
    # Populate the DataFrame with stock names
    stock_df = pd.DataFrame(index=range(num_rows), columns=range(num_cols))
    for i, stock in enumerate(stock_list):
        row = i // num_cols
        col = i % num_cols
        stock_df.iloc[row, col] = f'{i}: {stock}'
    
    # Display the stock list as a table
    print(tabulate(stock_df, headers='keys', tablefmt='grid', showindex=False, ))
    
    symbol = input('Enter stock symbol (or number), or type -1 to exit:\n')
    if symbol.isdigit() and int(symbol) in range(len(stock_list)):
        return stock_list[int(symbol)]
    elif symbol == '-1':
        return symbol
    elif symbol in stock_list:
        return symbol
    else:
        print('Invalid stock symbol. Please try again.')
        return prompt_user_for_stock(dataset_path)


if __name__ == '__main__':

    while True:
        dataset_path='../datasets/stock_data'
        symbol = prompt_user_for_stock(dataset_path)
        if symbol == '-1':
            print('Exiting...')
            break
        show_chart(symbol)