import cleanup
import cleaners as c

cleaning_functions = [c.example_clean_function()]

trained_data = cleanup.build_and_clean('data/train.csv', cleaning_functions)

