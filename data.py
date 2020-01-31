import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class DataReader:
    """Turns a CSV into all you need for scikit

        Args:
            filename(CSV): Input data
    """

    def __init__(self, filename):
       
        self.filename = filename
        self.df, self.data_train, self.data_test, self.target_train, \
        self.target_test = self.df_creator()
       
    def df_creator(self):
        """Creates a dataframe, removes NAN and splits to trainings
           sets from CVS-file.

           Args:

           Returns:
               dataframe, training and test sets.
        """

        df = pd.read_csv(self.filename)
        df.dropna(inplace=True, how="any")
        # print(df.shape)
        data = df[["pregnant", "glucose", "pressure",
                   "triceps", "insulin", "mass", "pedigree", "age"]]
        target = df[["diabetes"]]
        data_train, data_test, target_train, target_test = train_test_split(
            data, target, test_size=0.2)

        return df, data_train, data_test, target_train, target_test

    def scatter_plotter(self, dataseries1, dataseries2):
        """ Creates a scatter plot of 2 dataseries

            Args:
                dataseries1(String): Name of dataseries1
                dataseries2(String): Name of dataseries2

            Returns:
                Scatter Plot
        """
        df_to_scatter = self.df[[dataseries1, dataseries2, "diabetes"]]

        df_positive = df_to_scatter["diabetes"] == "pos"

        ax1 = df_to_scatter[df_positive].plot.scatter(
            x=dataseries1, y=dataseries2, c='Red', alpha=0.4, label="positive")

        ax2 = df_to_scatter[~df_positive].plot.scatter(
            x=dataseries1, y=dataseries2, c='Cyan', ax=ax1, alpha=0.4, label="negative")
        plt.show()


if __name__ == "__main__":
    diabetes_reader = DataReader("diabetes.csv")
    diabetes_reader.scatter_plotter("pregnant", "pressure")
