import pandas as pd


# I choose 50% agree because it contains all of the data and
# it makes it easier for us to identify the progress we make as we achieve fine-tuning the model.
# you also can try a different one.
# you can see in the folder FinancialPhraseBank-v1.0
# if you have any questions,please contact Moyu
def convert_to_csv():
    path = './FinancialPhraseBank-v1.0/Sentences_50Agree.txt'

    with open(path, 'r', encoding='latin-1') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        parts = line.strip().split('@')
        if len(parts) == 2:
            text = parts[0].strip()
            label = parts[1].strip()
            data.append([text, label])

    # dataframe
    df = pd.DataFrame(data, columns=['text', 'label'])

    df.to_csv('financial_phrase_bank.csv', index=False, encoding='utf-8')


def main():
    convert_to_csv()


if __name__ == "__main__":
    main()
