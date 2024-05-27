# MMCS-A-Multimodal-Model-Utilizing-Context-PDG-and-Simple-AST-for-Code-Summarization
1. convert every code to java files.

   `python code2Java.py`

2. use PropertyGraph-main to generate PDG.

   `$ cd out/artifacts/PropertyGraph_jar`
   `$ java -jar PropertyGraph.jar [-d <projectPath>] [-p] [-c] [-a]`
   `-d projectPath  
   -p: choose to generate PDG`
   `-c: choose to generate CFG`
   `-a: choose to generate AST`

   example: `java -jar PropertyGraph.jar -d test/src -p -c -a`

3. extract the dot file of PDG.

   `python dot_extract.py`

4. merge the PDG node and edge, get the jsonl file.

   `python merge2graphData.py`

5. covert the data to model input.

   `python Hu_dataset_process.py`

 6. train or test model.

    `python run.py`
