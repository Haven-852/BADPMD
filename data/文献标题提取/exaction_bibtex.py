import pandas as pd
import re


# Function to extract titles from .bib content
def extract_titles_from_bib(bib_content):
    # Regex pattern to match titles within the .bib file
    pattern = re.compile(r'title\s*=\s*{([^}]+)}', re.MULTILINE | re.DOTALL)
    # Find all matches for the pattern
    titles = pattern.findall(bib_content)
    # Remove any newline characters and leading/trailing spaces
    titles_cleaned = [title.replace('\n', ' ').strip() for title in titles]
    return titles_cleaned

# Path to the .bib file
bib_file_path = "..\\文献标题提取\\acm.bib"

# Read the .bib file content
with open(bib_file_path, 'r', encoding='utf-8') as bib_file:
    bib_content = bib_file.read()

# Extract titles from the .bib file content
titles = extract_titles_from_bib(bib_content)

# Create a DataFrame from the list of titles
df = pd.DataFrame(titles, columns=['Title'])

# Path to the CSV file we want to save
output_csv_path = "..\\文献标题提取\\titles.csv"

# Save the DataFrame to CSV
df.to_csv(output_csv_path, index=False)

output_csv_path
