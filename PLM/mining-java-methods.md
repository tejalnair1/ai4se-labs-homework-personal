
#Task: Build a dataset of Java methods
(This is a practice exercise -- you do not need to submit anything.)

This exercise is designed to give you hands-on practice in data collection and cleaning, which will make your life much easier when assignments get more complex (e.g., when DLLs or other advanced tasks are involved). It is intentionally lightweight.



## Instructions
For this bonus homework in AI4SE, your task is simple: **build a dataset of Java methods**.  

- **Collect methods** from any open-source projects you like.  
- Aim for **~20,000 methods for training** and **~5,000 methods for evaluation**.  
- Save the methods as **one or two CSV files** in whatever format you prefer.  
- Along with the CSV file(s), submit a **short description** (plain text is fine) that explains:
  - What data sources you used  
  - How you extracted the methods  
  - Any cleaning or filtering steps you applied  
  - Basic statistics (e.g., counts, method lengths, averages)  


---

## Example JSON-to-CSV Schema
While your submission should be in CSV, here is an example JSON structure that shows what each row might represent. You can flatten these fields into CSV columns.

```json
{
  "dataset_split": "train",
  "example_id": "awesome-lib@3f9c2b1:src/main/java/com/acme/math/Utils.java#120-162",
  "repo": {
    "name": "awesome-lib",
    "url": "https://github.com/acme/awesome-lib",
    "commit_sha": "3f9c2b1a7b0f41e3b7ad483f0b2f5e2f8f7e3a22",
    "license": "Apache-2.0"
  },
  "file": {
    "path": "src/main/java/com/acme/math/Utils.java",
    "language": "Java"
  },
  "method": {
    "name": "gcd",
    "qualified_name": "com.acme.math.Utils#gcd",
    "start_line": 120,
    "end_line": 162,
    "signature": "public static int gcd(int a, int b)",
    "original_code": "public static int gcd(int a, int b) {...}",
    "doc_comment": "/** Computes the greatest common divisor of two integers. */"
  },
  "code_tokens": [
    "public","static","int","gcd","(","int","a",",","int","b",")","{", "..."
  ]
}
```

---

## Example CSV Row (Flattened)

| dataset_split | repo_name  | repo_url                          | commit_sha | file_path                          | method_name | start_line | end_line | signature                           | original_code | code_tokens                                                                 |
|---------------|------------|-----------------------------------|------------|-----------------------------------|-------------|------------|----------|-------------------------------------|---------------|----------------------------------------------------------------------------|
| train         | awesome-lib| https://github.com/acme/awesome-lib | 3f9c2b1    | src/main/java/com/acme/math/Utils.java | gcd         | 120        | 162      | public static int gcd(int a, int b) | public static int gcd(int a, int b)... | public, static, int, gcd, (, int, a, , int, b, ), {, ... |

---

## Deliverables
1. **CSV file(s)**: one with 20k training examples, one with 5k evaluation examples (or combined with a `dataset_split` column).  
2. **Short description text**: include details of your process (sources, cleaning, statistics).  

---

## Quick Checklist
- [ ] Ensure exact commit SHA recorded per example.  
- [ ] Capture license; exclude repos with incompatible or missing licenses.  
- [ ] Record method boundaries (start/end lines) and signature.  
- [ ] Generate `code_tokens` consistently across train and eval.  
- [ ] Deduplicate and clean the data.
When cleaning the data, we intend removing all those cases, that are borderline and can hinder the training, for example:
  - Methods that are too short (e.g., less than 3 lines of code).
  - Methods that are too long (e.g., more than 100 lines of code).
  - Methods without any code (e.g., empty methods or methods with only comments).
  - Methods that are not valid Java code (e.g., due to parsing errors).
  - Methods from files that are not primarily Java files (e.g., mixed-language files).

