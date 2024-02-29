import pandas as pd
from lightweight_charts import Chart

if __name__ == '__main__':
    
    while True:
        try:
            symbol = input('Enter stock symbol: ')
            if symbol == '-1':
                break
            df = pd.read_csv(f'../datasets/stock_data/{symbol}.csv')
            chart = Chart()
            chart.set(df)
            chart.show(block=True)
        except:
            print('Invalid stock symbol. Please try again.')