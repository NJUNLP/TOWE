# Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling

This repository is under construction.

## Dependency:
* python3
* pytorch 0.4

## How to run (Take the 14res dataset as example)

1. (optional) prepare the word embeddings: (we have prepared for you in the code/data/ directory.)
    1. put the glove embeddings in the code/embedding directory.
    2. run the script:
    ```
    python build_vocab_embed.py --ds 14res
    ```
2. in the code/ directory:
    ```
    python main.py --ds 14res
    ```

