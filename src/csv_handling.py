class CsvHandler:
    def __init__(self, file_path, delimiter=",", encoding="utf-8"):
        self.file_path = file_path
        self.delimiter = delimiter
        self.encoding = encoding

    def peek_headers(self):
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                headers = f.readline().strip().split(self.delimiter)
                return headers
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def peek(self, num_lines=5):
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                headers = f.readline().strip().split(self.delimiter)
                preview_rows = []
                for _ in range(num_lines):
                    line = f.readline().strip()
                    if line:  # Check if line is not empty
                        preview_rows.append(line.split(self.delimiter))
            return headers, preview_rows
        except Exception as e:
            print(f"Error reading file: {e}")
            return None, None

    def drop_columns_by_name(self, columns_to_drop, output_path=None):
        if output_path is None:
            output_path = self.file_path

        try:
            # Read headers first
            with open(self.file_path, "r", encoding=self.encoding) as f:
                headers = f.readline().strip().split(self.delimiter)

            # Get indices of columns to keep
            keep_indices = [
                i for i, col in enumerate(headers) if col not in columns_to_drop
            ]

            if len(keep_indices) == len(headers):
                print("No columns found matching the names to drop")
                return False

            # Process the file
            with open(self.file_path, "r", encoding=self.encoding) as infile, open(
                output_path, "w", encoding=self.encoding
            ) as outfile:

                # Write new headers
                new_headers = [headers[i] for i in keep_indices]
                outfile.write(self.delimiter.join(new_headers) + "\n")

                # Process each line
                for line in infile:
                    if line.strip():  # Skip empty lines
                        values = line.strip().split(self.delimiter)
                        new_values = [values[i] for i in keep_indices]
                        outfile.write(self.delimiter.join(new_values) + "\n")

            return True

        except Exception as e:
            print(f"Error processing file: {e}")
            return False

    def get_shape(self):
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                # Get number of columns from header
                headers = f.readline().strip().split(self.delimiter)
                num_columns = len(headers)

                # Count rows (excluding header)
                num_rows = sum(1 for line in f if line.strip())

            return num_rows, num_columns

        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None, None

    def write_slice(self, output_file, row_slice=None, col_slice=None):
        try:
            with open(self.file_path, "r", encoding=self.encoding) as infile, open(
                output_file, "w", encoding=self.encoding
            ) as outfile:

                # Read and process header
                headers = infile.readline().strip().split(self.delimiter)
                if col_slice:
                    headers = headers[col_slice]
                outfile.write(self.delimiter.join(headers) + "\n")

                # Read all lines for row slicing
                lines = [line.strip() for line in infile if line.strip()]

                # Apply row slice if specified
                if row_slice:
                    lines = lines[row_slice]

                # Process selected rows
                for line in lines:
                    values = line.split(self.delimiter)
                    if col_slice:
                        values = values[col_slice]
                    outfile.write(self.delimiter.join(values) + "\n")

            return True

        except Exception as e:
            print(f"Error processing file slice: {e}")
            return False

    def split_into_parts(self, output_prefix, n_parts):
        try:
            # Get total number of rows
            total_rows = self.get_shape()[0]

            # Calculate rows per part (using integer division)
            rows_per_part = total_rows // n_parts

            # Split file into parts
            start_row = 0
            for i in range(n_parts - 1):
                output_file = f"{output_prefix}_{i+1}.csv"
                self.write_slice(
                    output_file,
                    row_slice=slice(start_row, start_row + rows_per_part),
                    col_slice=None,
                )

                start_row += rows_per_part

            # Write the last part with remaining rows
            output_file = f"{output_prefix}_{n_parts}.csv"
            self.write_slice(
                output_file, row_slice=slice(start_row, total_rows), col_slice=None
            )

            return True

        except Exception as e:
            print(f"Error splitting CSV file: {e}")
            return False

    @staticmethod
    def concatenate_files(input_files, output_file):
        """
        Concatenates multiple CSV files into a single output file.

        Args:
            input_files (list): List of paths to input CSV files
            output_file (str): Path to output CSV file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            first_header = None
            with open(output_file, "w") as outfile:
                # Process each input file
                for i, input_file in enumerate(input_files):
                    with open(input_file, "r") as infile:
                        # Read header
                        header = next(infile).strip()

                        # For first file, store header and write it
                        if i == 0:
                            first_header = header
                            outfile.write(header + "\n")
                        # For other files, verify header matches
                        elif header != first_header:
                            print("Error: CSV files have different headers")
                            return False

                        # Write remaining non-empty lines
                        for line in infile:
                            if line.strip():  # Skip empty lines
                                outfile.write(line)

            return True

        except Exception as e:
            print(f"Error concatenating CSV files: {e}")
            return False
