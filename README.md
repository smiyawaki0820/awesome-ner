# Named Entity Recognition
WIP

## Framework
- [transformers run_ner.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py)
- [spaCy EntityRecognizer](https://spacy.io/api/entityrecognizer)
  - [GiNZA - Japanese NLP Library](https://megagonlabs.github.io/ginza/) [[GitHub](https://github.com/megagonlabs/ginza)]
- [github.com/flairNLP/flair](https://github.com/flairNLP/flair)
- [Apache OpenNLP](https://opennlp.apache.org/docs/1.8.1/manual/opennlp.html#tools.namefind.recognition)
- [allennlp](https://guide.allennlp.org/common-architectures#3)
  - [himkt/allennlp-NER](https://github.com/himkt/allennlp-NER)

## Resource

### English

- [CoNLL-2002](https://www.clips.uantwerpen.be/conll2002/ner/) (Sang+'02)
- [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) (Sang and De Meulder+'03)
  - https://github.com/huggingface/datasets/tree/master/datasets/conll2003
- [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) (Weischedel+'13)
- [WNUT-2016](http://noisy-text.github.io/2016/ner-shared-task.html) (Strauss+'16)
- [WNUT-2017](http://noisy-text.github.io/2017/emerging-rare-entities.html) (Derczynski+Y17)
- [CREER](https://arxiv.org/abs/2204.12710)

### Japanese
|name|domain|all|train|valid|test|remarks|
|:---|:---:|---:|---:|---:|---:|:---|
|[Wikipediaを用いた日本語の固有表現抽出データセット - Stockmark](https://github.com/stockmarkteam/ner-wikipedia-dataset)|Wikipedia|||||[blog](https://tech.stockmark.co.jp/blog/202012_ner_dataset/)|
|[京都大学ウェブ文書リードコーパス](https://nlp.ist.i.kyoto-u.ac.jp/index.php?KWDLC)|web (various)|||||IREX|
|[UD Japanese GSD](https://universaldependencies.org/treebanks/ja_gsd/index.html)|Wikipedia|||||ENE|
|[GSK2014-A](https://www.gsk.or.jp/catalog/gsk2014-a/)|BCCWJ|||||ENE|
|[Contextual Clues for Named Entity Recognition in Japanese](https://cjki.org/samples/necc.htm)||||||
|[森羅](http://shinra-project.info/download/)|Wikipedia|||||

## Evaluation
- conlleval
  - http://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt
  - https://www.clips.uantwerpen.be/conll2000/chunking/output.html
- [seqeval](https://github.com/chakki-works/seqeval)
  - https://huggingface.co/spaces/evaluate-metric/seqeval

## LeaderBoard
- [ExplainaBoard](http://explainaboard.nlpedia.ai/leaderboard/task-ner/)

## Annontation Tools
- See: https://github.com/doccano/awesome-annotation-tools

## Active Learning
TBA

## Papers (entity-aware)
WIP

tag: 
- `seq`: Sequence Labeling
- `span`: Span-based
- `mrc`: Reading Comprehension

### preprint

### [AAAI2022](https://aaai-2022.virtualchair.net/papers.html?filter=titles&search=Entity)
- Li+'22 - Unified Named Entity Recognition as Word-Word Relation Classification (AAAI) [[paper](https://aaai-2022.virtualchair.net/poster_aaai742)]
- Agarwal+'22 - Towards Robust Named Entity Recognition via Temporal Domain Adaptation and Entity Context Understanding (AAAI) [[paper](https://aaai-2022.virtualchair.net/poster_dc136)]
- Shang+'22 - OneRel: Joint Entity and Relation Extraction with One Module in One Step (AAAI) [[paper](https://aaai-2022.virtualchair.net/poster_aaai246)]
- Raiman+'22 - DeepType 2: Superhuman Entity Linking, All You Need Is Type Interactions (AAAI) [[paper](https://aaai-2022.virtualchair.net/poster_aaai2612)]
- Liang+'22 - Exploring Entity Interactions for Few-Shot Relation Learning (AAAI) [[paper](https://aaai-2022.virtualchair.net/poster_sa368)]
- Xin+'22 - Ensemble Semi-Supervised Entity Alignment via Cycle-Teaching (AAAI) [[paper](https://aaai-2022.virtualchair.net/poster_aaai5065)]

### [TACL2022](https://aclanthology.org/events/tacl-2022/)
- Li+'22 - Ultra-fine Entity Typing with Indirect Supervision from Natural Language Inference (TACL) [[paper](https://aclanthology.org/2022.tacl-1.35/)]
- Cao+'22 - Multilingual Autoregressive Entity Linking (TACL) [[paper](https://aclanthology.org/2022.tacl-1.16/)]

### [LREC2022](https://lrec2022.lrec-conf.org/en/conference-programme/accepted-papers/)
- Yeshpanov+'22 - KazNERD: Kazakh Named Entity Recognition Dataset (LREC)
- Epure+'22 - Probing Pre-trained Auto-regressive Language Models for Named Entity Typing and Recognition (LREC)
- Jiang+'22 - Annotating the Tweebank Corpus on Named Entity Recognition and Building NLP Models for Social Media Analysis (LREC)
- Phan+'22 - A Named Entity Recognition Corpus for Vietnamese Biomedical Texts to Support Tuberculosis Treatment (LREC)
- Green+'22 - Development of a Benchmark Corpus to Support Entity Recognition in Job Descriptions (LREC)
- Aprosio+'22 - KIND: an Italian Multi-Domain Dataset for Named Entity Recognition (LREC)
- Rückert+'22 - A Unified Approach to Entity-Centric Context Tracking in Social Conversations (LREC)
- Jarrar+'22 - Wojood: Nested Arabic Named Entity Corpus and Recognition using BERT (LREC)
- Strobl+'22 - Enhanced Entity Annotations for Multilingual Corpora (LREC)
- Orasmaa+'22 - Named Entity Recognition in Estonian 19th Century Parish Court Records (LREC)
- Severini+'22 - Towards a Broad Coverage Named Entity Resource: A Data-Efficient Approach for Many Diverse Languages (LREC)
- Garcia-Duran+'22 - Efficient Entity Candidate Generation for Low-Resource Languages (LREC)
- Bonet+'22 - Spanish Datasets for Sensitive Entity Detection in the Legal Domain (LREC)
- Pathak+'22 - AsNER - Annotated Dataset and Baselines for Assamese Named Entity recognition (LREC)
- Hatab+'22 - Enhancing Deep Learning with Embedded Features for Arabic Named Entity Recognition (LREC)
- Skórzewski+'22 - Named Entity Recognition to Detect Criminal Texts on the Web (LREC)
- Ivanova+'22 - Comparing Annotated Datasets for Named Entity Recognition in English Literature (LREC)
- Loukachevitch+'22 - Entity Linking over Nested Named Entities for Russian (LREC)
- Murthy+'22 - HiNER: A large Hindi Named Entity Recognition Dataset (LREC)
- Cui+'22 - OpenEL: An Annotated Corpus for Entity Linking and Discourse in Open Domain Dialogue (LREC)
- Sun+'22 - SPORTSINTERVIEW: A Large-Scale Sports Interview Benchmark for Entity-centric Dialogues (LREC)
- Alekseev+'22 - Medical Crossing: a Cross-lingual Evaluation of Clinical Entity Linking (LREC)
- Mullick+'22 - Using Sentence-level Classification Helps Entity Extraction from Material Science Literature (LREC)
- Çarık +'22 - A Twitter Corpus for Named Entity Recognition in Turkish (LREC)
- Okur+'22 - Exploring Paraphrase Generation and Entity Extraction for Multimodal Dialogue System (LREC)

### [NAACL2022](https://2022.naacl.org/program/accepted_papers/)
- Wu+'22 - Robust Self-Augmentation for Named Entity Recognition with Meta Reweighting (NAACL) [[arXiv](https://arxiv.org/abs/2204.11406)]
- Fetahu+'22 - Dynamic Gazetteer Integration in Multilingual Models for Cross-Lingual and Cross-Domain Named Entity Recognition (NAACL) [[AmazonScience](https://assets.amazon.science/4b/72/7d7551b144039c3f9b955b8e6cdc/dynamic-gazetteer-integration-in-multilingual-models-for-cross-lingual-and-cross-domain-named-entity-recognition.pdf)]
- Wang+'22 - Sentence-Level Resampling for Named Entity Recognition (NAACL) [[openreview](https://openreview.net/pdf?id=W_0vwhG1gkW)]
- Hu+'22 - Hero-Gang Neural Model For Named Entity Recognition (NAACL) [[arXiv](https://arxiv.org/abs/2205.07177)]
- Pasad+'22 - On the Use of External Data for Spoken Named Entity Recognition (NAACL) [[arXiv](https://arxiv.org/abs/2112.07648)]
- Ying+'22 - Label Refinement via Contrastive Learning for Distantly-Supervised Named Entity Recognition (NAACL)
- Gu+'22 - Delving Deep into Regularity: A Simple but Effective Method for Chinese Named Entity Recognition (NAACL) [[arXiv](http://arxiv.org/abs/2204.05544)]
- Shao+'22 - Low-resource Entity Set Expansion: A Comprehensive Study on User-generated Text (NAACL) [[acsweb](https://acsweb.ucsd.edu/~yshao/Low_resource_entity_set_expans.pdf)]
- Tedeschi+'22 - MultiNER: A Multilingual, Multi-Genre and Fine-Grained Dataset for Named Entity Recognition (NAACL)
- Shrimal+'22 - NER-MQMRC: Formulating Named Entity Recognition as Multi Question Machine Reading Comprehension (NAACL) [[arXiv](https://arxiv.org/abs/2205.05904)]
- Agarwal+'22 - Entity Linking via Explicit Mention-Mention Coreference Modeling (NAACL) [[openreview](https://openreview.net/pdf?id=f_1MD91kBza)]
- Wang+'22 - Should We Rely on Entity Mentions for Relation Extraction? Debiasing Relation Extraction with Counterfactual Analysis (NAACL) [[arXiv](https://arxiv.org/abs/2205.03784)]
- Hakami+'22 - Learning to Borrow– Relation Representation for Without-Mention Entity-Pairs for Knowledge Graph Completion (NAACL) [[arXiv](https://arxiv.org/abs/2204.13097)]
- Xu+'22 - Modeling Explicit Task Interactions in Document-Level Joint Entity and Relation Extraction (NAACL) [[arXiv](https://arxiv.org/abs/2205.01909)]
- Yuan+'22 - Generative Biomedical Entity Linking via Knowledge Base-Guided Pre-training and Synonyms-Aware Fine-tuning (NAACL) [[arXiv](https://arxiv.org/abs/2204.05164)]
- Chai+'22 - Incorporating Centering Theory into Entity Coreference Resolution (NAACL) 
- Yu+'22 - Relation-Specific Attentions over Entity Mentions for Enhanced Document-Level Relation Extraction [[arXiv](https://arxiv.org/abs/2205.14393)]
- Bhargav+'22 - Zero-shot Entity Linking with Less Data (NAACL) [[arXiv](https://arxiv.org/abs/1906.07348)]
- Liu+'22 - Dangling-Aware Entity Alignment with Mixed High-Order Proximities (NAACL) [[arXiv](https://aps.arxiv.org/abs/2205.02406)]
- Ayoola+'22 - ReFinED: An Efficient Zero-shot-capable Approach to End-to-End Entity Linking (NAACL) [[AmazonScience](https://www.amazon.science/publications/refined-an-efficient-zero-shot-capable-approach-to-end-to-end-entity-linking)]
- Laskar+'22 - BLINK with Elasticsearch for Efficient Entity Linking in Business Conversations (NAACL) [[arXiv](https://arxiv.org/abs/2205.04438)]
- Nishikawa+'22 - EASE: Entity-Aware Contrastive Learning of Sentence Embedding  (NAACL) [[arXiv](https://arxiv.org/abs/2205.04260)]
- Ayoola+'22 - Improving Entity Disambiguation by Reasoning over a Knowledge Base (NAACL) [[AmazonScience](https://assets.amazon.science/26/9a/31bb24de49cbadbf779976806991/improving-entity-disambiguation-by-reasoning-over-a-knowledge-base.pdf)]
- Yan+'22 - On the Robustness of Reading Comprehension Models to Entity Renaming (NAACL) [[arXiv](https://arxiv.org/abs/2110.08555)]
- Yamada+'22 - Global Entity Disambiguation with BERT [[arXiv](https://arxiv.org/abs/1909.00426)]
- Onoe+'22 - Entity Cloze By Date: Understanding what LMs know about unseen entities (NAACL) [[arXiv](https://arxiv.org/abs/2205.02832)]
- Deeksha+'22 - Commonsense and Named Entity Aware Knowledge Grounded Dialogue Generation (NAACL) [[arXiv](https://arxiv.org/abs/2205.13928)]
- Schuster+'22 - When a sentence does not introduce a discourse entity, Transformer-based models still often refer to it (NAACL) [[arXiv](https://arxiv.org/abs/2205.03472)]
- Zhang+'22 - Improving the Faithfulness of Abstractive Summarization via Entity Coverage Control (NAACL)
- Wang+'22 - ITA: Image-Text Alignments for Multi-Modal Named Entity Recognition (NAACL) [[arXiv](https://arxiv.org/abs/2112.06482)]
- Chen+'22 - Good Visual Guidance Make A Better Extractor: Hierarchical Visual Prefix for Multimodal Entity and Relation Extraction (NAACL) [[paperswithcode](https://paperswithcode.com/paper/good-visual-guidance-makes-a-better-extractor)]


### [ACL2022](https://aclanthology.org/events/acl-2022/)
- Wang+'22 - MINER: Improving Out-of-Vocabulary Named Entity Recognition from an Information Theoretic Perspective (ACL) [[paper](https://aclanthology.org/2022.acl-long.383/)]
- Ma+'22 - Decomposed Meta-Learning for Few-Shot Named Entity Recognition (ACL) [[paper](https://aclanthology.org/2022.findings-acl.124/)]
- Li+'22 - An Unsupervised Multiple-Task and Multiple-Teacher Model for Cross-lingual Named Entity Recognition (ACL) [[paper](https://aclanthology.org/2022.acl-long.14/)]
- Yang+'22 - Bottom-Up Constituency Parsing and Nested Named Entity Recognition with Pointer Networks (ACL) [[paper](https://aclanthology.org/2022.acl-long.171/)]
- Zhu+'22 - Boundary Smoothing for Named Entity Recognition (ACL) [[paper](https://aclanthology.org/2022.acl-long.490/)]
- Snigdha+'22 - CONTaiNER: Few-Shot Named Entity Recognition via Contrastive Learning (ACL) [[paper](https://aclanthology.org/2022.acl-long.439/)]
- Zhou+'22 - Distantly Supervised Named Entity Recognition via Confidence-Based Multi-Class Positive and Unlabeled Learning (ACL) [[paper](https://aclanthology.org/2022.acl-long.498/)]
- Wang+'22 - Few-Shot Class-Incremental Learning for Named Entity Recognition (ACL) [[paper](https://aclanthology.org/2022.acl-long.43/)]
- Chen+'22 - Few-shot Named Entity Recognition with Self-describing Networks (ACL) [[paper](https://aclanthology.org/2022.acl-long.392/)]
- Lou+'22 - Nested Named Entity Recognition as Latent Lexicalized Constituency Parsing (ACL) [[paper](https://aclanthology.org/2022.acl-long.428/)]
- Wan+'22 - Nested Named Entity Recognition with Span-level Graphs (ACL) [[paper](https://aclanthology.org/2022.acl-long.63/)]
- Shen+'22 - Parallel Instance Query Network for Named Entity Recognition (ACL) [[paper](https://aclanthology.org/2022.acl-long.67/)]
- Zheng+'22 - Cross-domain Named Entity Recognition via Graph Matching (ACL) [[paper](https://aclanthology.org/2022.findings-acl.210/)]
- Huang+'22 - Extract-Select: A Span Selection Framework for Nested Named Entity Recognition with Generative Adversarial Training (ACL) [[paper](https://aclanthology.org/2022.findings-acl.9/)]
- Yuan+'22 - Fusing Heterogeneous Factors with Triaffine Mechanism for Nested Named Entity Recognition (ACL) [[paper](https://aclanthology.org/2022.findings-acl.250/)]
- Ma+'22 - Label Semantics for Few Shot Named Entity Recognition (ACL) [[paper](https://aclanthology.org/2022.findings-acl.155/)]
- Xia+'22 - Learn and Review: Enhancing Continual Named Entity Recognition via Reviewing Synthetic Samples (ACL) [[paper](https://aclanthology.org/2022.findings-acl.179/)]
- Buaphet+'22 - Thai Nested Named Entity Recognition Corpus (ACL) [[paper](https://aclanthology.org/2022.findings-acl.116/)]
- Reich+'22 - Leveraging Expert Guided Adversarial Augmentation For Improving Generalization in Named Entity Recognition (ACL) [[paper](https://aclanthology.org/2022.findings-acl.154/)]
- Loukas+'22 - FiNER: Financial Numeric Entity Recognition for XBRL Tagging (ACL) [[paper](https://aclanthology.org/2022.acl-long.303/)]
- Zhou+'22 - MELM: Data Augmentation with Masked Entity Language Modeling for Low-Resource NER (ACL) [[paper](https://aclanthology.org/2022.acl-long.160/)]
- Ri+'22 - mLUKE: The Power of Entity Representations in Multilingual Pretrained Language Models (ACL) [[paper](https://aclanthology.org/2022.acl-long.505/)]
- Leszczynski+'22 - TABi: Type-Aware Bi-Encoders for Open-Domain Entity Retrieval (ACL) [[paper](https://aclanthology.org/2022.findings-acl.169/)]
- Galperin+'22 - Cross-Lingual UMLS Named Entity Linking using UMLS Dictionary Fine-Tuning (ACL) [[paper](https://aclanthology.org/2022.findings-acl.266/)]
- Han+'22 - Cross-Lingual Contrastive Learning for Fine-Grained Entity Typing for Low-Resource Languages (ACL) [[paper](https://aclanthology.org/2022.acl-long.159/)]
- Pang+'22 - Divide and Denoise: Learning from Noisy Labels in Fine-Grained Entity Typing with Cluster-Wise Loss Correction (ACL) [[paper](https://aclanthology.org/2022.acl-long.141/)]
- Chen+'22 - Learning from Sibling Mentions with Scalable Graph Inference in Fine-Grained Entity Typing (ACL) [[paper](https://aclanthology.org/2022.acl-long.147/)]
- Wang+'22 - WikiDiverse: A Multimodal Entity Linking Dataset with Diversified Contextual Topics and Entity Types (ACL) [[paper](https://aclanthology.org/2022.acl-long.328/)]
- Zaporojets+'22 - Towards Consistent Document-level Entity Linking: Joint Models for Entity Linking and Coreference Resolution (ACL) [[paper](https://aclanthology.org/2022.acl-short.88/)]
- Sun+'22 - A Transformational Biencoder with In-Domain Negative Sampling for Zero-Shot Entity Linking (ACL) [[paper](https://aclanthology.org/2022.findings-acl.114/)]
- Mrini+'22 - Detection, Disambiguation, Re-ranking: Autoregressive Entity Linking as a Multi-Task Problem (ACL) [[paper](https://aclanthology.org/2022.findings-acl.156/)]
- Jin+'22 - How Can Cross-lingual Knowledge Contribute Better to Fine-Grained Entity Typing? (ACL) [[paper](https://aclanthology.org/2022.findings-acl.243/)]
- Lai+'22 - Improving Candidate Retrieval with Entity Profile Generation for Wikidata Entity Linking (ACL) [[paper](https://aclanthology.org/2022.findings-acl.292/)]

## References
- [A 2020 Guide to Named Entity Recognition - LaptrinhX / News](https://laptrinhx.com/news/a-2020-guide-to-named-entity-recognition-W7eeD6j/)
- [Named Entity Recognition - Nanonets](https://nanonets.com/blog/named-entity-recognition-with-nltk-and-spacy/)
- [David S. Batista - Named-Entity evaluation metrics based on entity-level (Blog)](https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/)

### Projects
- [森羅 SINRA](http://shinra-project.info/shinra2020ml/overview/)
- [拡張固有表現階層](http://liat-aip.sakura.ne.jp/ene/ene8/definition_jp/html/home.html)
<img src="http://ene-project.info/wp-content/uploads/sites/13/2020/09/zentaizu8-0_J_E_200925_J-1024x576.png" title="拡張固有表現 全体図 (ver.8.0)" alt="http://ene-project.info/ene8/">

### Tips
TBA
