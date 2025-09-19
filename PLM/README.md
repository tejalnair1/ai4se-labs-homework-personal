# N-Gram Code Recommender

This project builds an **N-gram language model** over Java method source code, using data collected from an open-source repository.

## Project Structure

- **`dataset_java_methods.ipynb`**  
  Extracts Java methods and prepares the dataset.  
  **Steps included in the notebook:**
  1. **Repository Mining**: Crawl through Java project directories to collect `.java` files.  
  2. **Method Extraction**: Parse source files using `javalang` to get individual method bodies.  
  3. **Preprocessing**: Clean method bodies (remove comments, imports, etc.).  
  4. **CSV Construction**: Save extracted methods into `methods.csv` with columns such as:  
     - `method_code`: the raw Java method body.  
     - `dataset_split`: randomized split into ~25,000 train and remaining test.  
  5. **Output**: Running this notebook produces the dataset csv file:  
     ```
     methods.csv
     ```

- **`n-gram-recommender.ipynb`**  
  Trains and evaluates an **N-gram language model** on the `methods.csv` dataset.  
  **Steps included in the notebook:**
  1. **Load Dataset**: Read `methods.csv` and split into train/test based on the `dataset_split` column.  
  2. **Preprocessing**:  
     - Remove comments, `import`/`package` lines.  
     - Normalize string/char tokens to single token.  
     - Tokenize into identifiers, numbers, operations, and punctuation.  
     - Add `<BOS>`/`<EOS>` markers.  
  3. **Model Training**:  
     - Build an **N-gram LM** (N can be changed).  
     - Use add-k smoothing.  
     - Maintain vocab and n-gram counts.  
  4. **Evaluation**:  
     - Compute perplexity on the test split.  
     - Compare results across different N.  
     - Report vocab size and test perplexity.  
  5. **Prediction Demo**:  
     - Generate top-k next-token predictions given a fixed word (`public`).  
     - Sample continuation sequences from the trained model.  

## Usage

1. Run **`dataset_java_methods.ipynb`** to produce the dataset:
   ```bash
   methods.csv
   ```

2. Run **`n-gram-recommender.ipynb`** to train and evaluate the N-gram recommender.  
