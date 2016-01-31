# !/bin/bash
# Generates a template for a cleaning function to be used to clean data in the data directory

echo "Generating cleaner:" $1

array=("" "def $1():" "    def $1(dataframe):" "        return dataframe" "    return $1")
for i in "${array[@]}"; do   # The quotes are necessary here
    printf '%s\n' "$i">> cleaners.py
done
