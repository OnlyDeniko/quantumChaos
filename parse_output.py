import pandas as pd
import click

@click.command()
@click.option('--output', default='output.xlsx', help='path to save output results')
@click.option('--sheet',  default='Sheet1',      help='name of sheet in output results')
def parse(output, sheet):
    frames = []

    for iteration in range(2, 17):
        n = int(input())
        eigenvalues = list()
        for i in range(n):
            elem = tuple(map(float, input().split()))
            eigenvalues.append(elem)

        dk = list()
        for i in range(n):
            elem = tuple(map(float, input().split()))
            dk.append(elem)

        uk_name = 'u_k_' + str(n)
        dk_name = 'd_k_' + str(n)
        tmp = pd.DataFrame({
            uk_name: eigenvalues,
            dk_name: dk})
        frames.append(tmp)

    pd.concat(frames, axis=1).to_excel(output, sheet_name=sheet, engine='xlsxwriter')

if __name__ == '__main__':
    parse()