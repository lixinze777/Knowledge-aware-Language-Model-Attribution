# Towards Verifiable Generation: A Benchmark for Knowledge-aware Language Model Attribution
Although achieving great success, Large Language Models (LLMs) usually suffer from unreliable hallucinations. Although language attribution can be a potential solution, there are no suitable benchmarks and evaluation metrics to attribute LLMs to structured knowledge. In this paper, we define a new task of Knowledge-aware Language Model Attribution (KaLMA) that improves upon three core concerns with conventional attributed LMs. First, we extend attribution source from unstructured texts to Knowledge Graph (KG), whose rich structures benefit both the attribution performance and working scenarios. Second, we propose a new "Conscious Incompetence" setting considering the incomplete knowledge repository, where the model identifies the need for supporting knowledge beyond the provided KG. Third, we propose a comprehensive automatic evaluation metric encompassing text quality, citation quality, and text citation alignment. To implement the above innovations, we build a dataset in biography domain BioKaLMA via evolutionary question generation strategy, to control the question complexity and necessary knowledge to the answer. For evaluation, we develop a baseline solution and demonstrate the room for improvement in LLMs' citation generation, emphasizing the importance of incorporating the "Conscious Incompetence" setting, and the critical role of retrieval accuracy.

## Paper Link
[Download Paper](https://aclanthology.org/2024.findings-acl.28.pdf)

## Citation
Please cite our paper if you use KaLMA in your work:
```bibtex
@inproceedings{DBLP:conf/acl/Li0PMS24,
  author       = {Xinze Li and
                  Yixin Cao and
                  Liangming Pan and
                  Yubo Ma and
                  Aixin Sun},
  editor       = {Lun{-}Wei Ku and
                  Andre Martins and
                  Vivek Srikumar},
  title        = {Towards Verifiable Generation: {A} Benchmark for Knowledge-aware Language
                  Model Attribution},
  booktitle    = {Findings of the Association for Computational Linguistics, {ACL} 2024,
                  Bangkok, Thailand and virtual meeting, August 11-16, 2024},
  pages        = {493--516},
  publisher    = {Association for Computational Linguistics},
  year         = {2024},
  url          = {https://doi.org/10.18653/v1/2024.findings-acl.28},
  doi          = {10.18653/V1/2024.FINDINGS-ACL.28},
  timestamp    = {Tue, 24 Sep 2024 10:55:33 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/Li0PMS24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
