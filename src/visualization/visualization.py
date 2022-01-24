import matplotlib.pyplot as plt


class Visualization:

    @classmethod
    def plot_column_by_target(cls, df, column, target):
        grouped_df = df.groupby(target).count()
        column_count_per_target = grouped_df[column].to_numpy()
        unique_target_values = grouped_df.index.to_numpy()

        plt.yticks(column_count_per_target)
        plt.xticks(unique_target_values)
        plt.bar(unique_target_values, column_count_per_target, align='center')
        plt.ylabel('Count')
        plt.xlabel('Label')
        plt.title('Label Distribution across reviews')