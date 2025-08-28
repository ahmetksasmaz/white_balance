import numpy as np
from configuration import *

class Evaluator:
    def __init__(self, results_filename, evaluation_filename, export_period = 100):
        self.results_filename = results_filename
        self.evaluation_filename = evaluation_filename
        self.results = {}
        self.evaluation = {}
        self.export_period = export_period
        self.__updated_result_count = 0

        self.__parse_results()

    def __parse_results(self):
        file = open(self.results_filename, "r")
        for line in file:
            # dataset_name, image_name, algorithm, error_type, value
            dataset_name, image_name, algorithm, error_type, value = line.strip().split(",")
            value = float(value)
            if dataset_name not in self.results:
                self.results[dataset_name] = {}
            if image_name not in self.results[dataset_name]:
                self.results[dataset_name][image_name] = {}
            if algorithm not in self.results[dataset_name][image_name]:
                self.results[dataset_name][image_name][algorithm] = {}
            self.results[dataset_name][image_name][algorithm][error_type] = value

        file.close()

    def __export_results(self):
        file = open(self.results_filename, "w")
        for dataset_name in self.results:
            for image_name in self.results[dataset_name]:
                for algorithm in self.results[dataset_name][image_name]:
                    for error_type in self.results[dataset_name][image_name][algorithm]:
                        value = self.results[dataset_name][image_name][algorithm][error_type]
                        file.write(f"{dataset_name},{image_name},{algorithm},{error_type},{value}\n")
        file.close()

    def evaluate(self):
        # Create evaluation structure
        for dataset in VALID_DATASETS:
            if dataset not in self.evaluation:
                self.evaluation[dataset] = {}
            for algorithm in VALID_ALGORITHMS:
                if algorithm not in self.evaluation[dataset]:
                    self.evaluation[dataset][algorithm] = {}
                for error_type in VALID_ERROR_METRICS:
                    self.evaluation[dataset][algorithm][error_type] = []
        # Iterate over results
        for dataset_name in self.results:
            for image_name in self.results[dataset_name]:
                for algorithm in self.results[dataset_name][image_name]:
                    for error_type in self.results[dataset_name][image_name][algorithm]:
                        self.evaluation[dataset_name][algorithm][error_type].append(self.results[dataset_name][image_name][algorithm][error_type])
        self.__export_results()
        self.__export_evaluation()

    def __export_evaluation(self):
        evaluation_file = open(self.evaluation_filename, "w")
        # Compute mean and standart deviation
        for dataset in VALID_DATASETS:
            if dataset not in self.evaluation:
                self.evaluation[dataset] = {}
            for algorithm in VALID_ALGORITHMS:
                if algorithm not in self.evaluation[dataset]:
                    self.evaluation[dataset][algorithm] = {}
                for error_type in VALID_ERROR_METRICS:
                    values = self.evaluation[dataset][algorithm][error_type]
                    mean = sum(values) / len(values) if values else 0.0
                    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5 if values else 0.0
                    evaluation_file.write(f"{dataset},{algorithm},{error_type},{mean},{std}\n")
        evaluation_file.close()

    def has_processed(self, dataset_name, image_name, algorithm):
        if dataset_name in self.results:
            if image_name in self.results[dataset_name]:
                if algorithm in self.results[dataset_name][image_name]:
                    return True
        return False

    def update_processed(self, dataset_name, image_name, algorithm, errors):
        if dataset_name not in self.results:
            self.results[dataset_name] = {}
        if image_name not in self.results[dataset_name]:
            self.results[dataset_name][image_name] = {}
        if algorithm not in self.results[dataset_name][image_name]:
            self.results[dataset_name][image_name][algorithm] = {}
        for error in errors:
            value = errors[error]
            self.results[dataset_name][image_name][algorithm][error] = value
        self.__updated_result_count += 1
        if self.__updated_result_count >= self.export_period:
            self.evaluate()
            self.__updated_result_count = 0
