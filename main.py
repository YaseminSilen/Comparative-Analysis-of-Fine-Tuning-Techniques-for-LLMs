import convert2csv
import preprocessing_FPB
import preprocessing_tfns


def main():
    # the format of financial phrase bank is txt, so it needs to be converted to csv first.
    convert2csv.main()
    # preprocessing for financial phrase bank
    preprocessing_FPB.main()
    # preprocessing for twitter-financial-news-sentiment
    preprocessing_tfns.main()


if __name__ == "__main__":
    main()
