import pandas as pd
import numpy as np
import time


class DocumentUnit:
    """
    Object that document the output from the model in an epoch
    """

    def __init__(self, columns):
        self.data_dict = dict([(col, []) for col in columns])
        # skip case level pred and labels since they has 1/2 length
        self.skip_key = ["case_pred", "case_label", "left_case_pred",
                         "right_case_pred", "fusion_case_pred", "left_right_case_pred"]
        # accumulator for localization
        self.localization_accumulator = None

    def update_accumulator(self, delta):
        """
        Method that accumulates localization pixel-wise values for metrics such as mIOU
        :param delta:
        :return:
        """
        if self.localization_accumulator is None:
            self.localization_accumulator = delta
        else:
            assert self.localization_accumulator.shape == delta.shape,\
                "self.localization_accumulator.shape {0} != delta.shape {1}".format(self.localization_accumulator.shape, delta.shape)
            self.localization_accumulator += delta

    def add_values(self, column, values, process_method=lambda x: x):
        """
        Method that add values into the document unit
        :param column:
        :param values:
        :return:
        """
        for val in values:
            self.data_dict[column].append(process_method(val))

    def form_df(self):
        """
        Method that creates a dataframe out of stored data
        :return:
        """
        to_be_save_dict = {}
        for key in self.data_dict:
            if key not in self.skip_key and len(self.data_dict[key]) != 0:
                to_be_save_dict[key] = self.data_dict[key]
        df = pd.DataFrame(to_be_save_dict).reset_index()
        return df

    def get_latest_results(self):
        """
        Method that retrieves the latest results from the stored data
        :return:
        """
        to_be_save_dict = {}
        for key in self.data_dict:
            if key not in self.skip_key and len(self.data_dict[key]) != 0:
                to_be_save_dict[key] = self.data_dict[key][-1]
        return to_be_save_dict

    def to_csv(self, dir):
        """
        Export to csv
        :param dir:
        :return:
        """
        df = self.form_df()
        df.to_csv(dir, index=False)



class RuntimeProfiler:
    """
    Object that documents run-time
    """
    def __init__(self):
        self.elpased_time_dict = {}
        self.current_time_point = None

    def tik(self, time_category=None):
        """
        Take a time point
        :param time_category:
        :return:
        """
        new_time = time.time()
        return_time = False
        if self.current_time_point is not None:
            return_time = True
            elapsed_time = new_time - self.current_time_point
            if time_category is not None:
                if time_category not in self.elpased_time_dict:
                    self.elpased_time_dict[time_category] = []
                self.elpased_time_dict[time_category].append(elapsed_time)
        self.current_time_point = new_time
        if return_time:
            return elapsed_time


    def report_avg(self):
        """
        Generate a format string for average run-time statistics
        :return:
        """
        output_str = ""
        for time_category in self.elpased_time_dict:
            output_str += "category:{0}, avg_time:{1}, std_time:{2}, min_time:{3}, max_time:{4}, num_points:{5}\n".format(
                time_category, np.mean(self.elpased_time_dict[time_category]), np.std(self.elpased_time_dict[time_category]),
                np.min(self.elpased_time_dict[time_category]), np.max(self.elpased_time_dict[time_category]),
                len(self.elpased_time_dict[time_category])
            )
        return output_str

    def report_latest(self):
        """
        Generate a format string for the latest run-time statistics
        :return:
        """
        output_str = ""
        for time_category in self.elpased_time_dict:
            output_str += "category:{0}, runtime:{1} \n".format(time_category, self.elpased_time_dict[time_category][-1])
        return output_str