## Benchmark comparison

Checkpoint **379079** was evaluated unchanged on physical TPUs using the complete pinned benchmark inventories: English MTEB (41 tasks), multilingual MTEB (131 tasks), CoIR (10 tasks), and MIRACL hard-negative retrieval (18 language subsets).

| Benchmark | GOAT mmBERT YAT Base, MLM-only | Previous YAT embedding v1, fine-tuned | mmBERT-base paper, embedding fine-tuned |
|---|---:|---:|---:|
| English MTEB, seven-category mean | 36.84 | 52.78 | 53.9 |
| Multilingual MTEB, eight-category mean | 31.77 | 48.21 | 54.1 |
| Multilingual MTEB, native nine-category mean | 28.07 | 42.40 | — |
| CoIR, ten-task mean | 9.91 | 43.45 | 42.2 |
| MIRACL hard-negative, 18-language mean | 1.82 | 46.37 | — |

Evaluation used MTEB 2.21.8, length512, right truncation, mean pooling over nonpadding tokens including special tokens, FP32 L2 normalization, no prompts, and inference batch16. Reported values use the 0–100 display scale; retained receipts preserve native main scores. The eight-category multilingual mean follows the paper table; the native nine-category mean also includes InstructionReranking.

The previous-model column uses the [published full-suite receipts](https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1/tree/571d0ef3d8e1381055845fce334a48be253c3fea/evaluation) with the same pinned task/dataset revisions and split/subset coverage. Its model used contrastive fine-tuning with MS MARCO, MIRACL and CodeSearchNet pairs. The [mmBERT paper](https://arxiv.org/html/2509.06888v1), Tables4–6, reports embedding results after MS MARCO fine-tuning and learning-rate selection. This GOAT checkpoint is an MLM-only base model.

[Full task, category and language results](GOAT_BENCHMARKS.md) and the complete machine-readable receipts are retained in the [published model evaluation directory](https://huggingface.co/mlnomad/goat-mmbert-yat-base-mlm-10b/tree/336c24b33c84305adf4b3fb152a588efe68b2b91/evaluation).

Measurements combine authenticated receipts from three evaluation executions. Each accepted job retains its original execution identity, frozen protocol and final unchanged-weight proof. MIRACL language subsets and WebLINX splits are included once each.


### Category scores

| Suite | Category | GOAT base | Previous published |
|---|---|---:|---:|
| english | Classification | 52.89 | 66.32 |
| english | Clustering | 34.84 | 40.95 |
| english | PairClassification | 47.74 | 78.33 |
| english | Reranking | 36.12 | 43.63 |
| english | Retrieval | 5.10 | 42.96 |
| english | STS | 49.45 | 74.70 |
| english | Summarization | 31.72 | 22.58 |
| multilingual | BitextMining | 12.66 | 43.99 |
| multilingual | Classification | 45.20 | 50.57 |
| multilingual | Clustering | 32.71 | 37.96 |
| multilingual | InstructionReranking | -1.54 | -4.07 |
| multilingual | MultilabelClassification | 13.85 | 15.07 |
| multilingual | PairClassification | 64.59 | 73.63 |
| multilingual | Reranking | 31.83 | 55.66 |
| multilingual | Retrieval | 11.01 | 44.33 |
| multilingual | STS | 42.27 | 64.45 |
| coir | Retrieval | 9.91 | 43.45 |

### Task scores

| Suite | Task | GOAT base | Previous published |
|---|---|---:|---:|
| english | AmazonCounterfactualClassification | 69.88 | 60.10 |
| english | ArXivHierarchicalClusteringP2P | 53.23 | 55.39 |
| english | ArXivHierarchicalClusteringS2S | 50.33 | 51.64 |
| english | ArguAna | 17.04 | 45.12 |
| english | AskUbuntuDupQuestions | 46.41 | 56.52 |
| english | BIOSSES | 47.63 | 76.12 |
| english | Banking77Classification | 38.00 | 74.03 |
| english | BiorxivClusteringP2P.v2 | 29.97 | 32.91 |
| english | CQADupstackGamingRetrieval | 4.51 | 46.97 |
| english | CQADupstackUnixRetrieval | 4.48 | 29.03 |
| english | ClimateFEVERHardNegatives | 3.30 | 21.82 |
| english | FEVERHardNegatives | 0.89 | 68.66 |
| english | FiQA2018 | 0.79 | 25.32 |
| english | HotpotQAHardNegatives | 4.11 | 51.77 |
| english | ImdbClassification | 59.93 | 60.67 |
| english | MTOPDomainClassification | 66.63 | 86.62 |
| english | MassiveIntentClassification | 36.51 | 64.28 |
| english | MassiveScenarioClassification | 46.29 | 71.60 |
| english | MedrxivClusteringP2P.v2 | 29.38 | 33.91 |
| english | MedrxivClusteringS2S.v2 | 25.61 | 30.38 |
| english | MindSmallReranking | 25.83 | 30.73 |
| english | SCIDOCS | 0.72 | 10.59 |
| english | SICK-R | 57.94 | 69.47 |
| english | STS12 | 36.61 | 71.03 |
| english | STS13 | 47.17 | 77.23 |
| english | STS14 | 40.18 | 72.83 |
| english | STS15 | 54.63 | 81.90 |
| english | STS17 | 60.44 | 83.98 |
| english | STS22.v2 | 57.45 | 65.67 |
| english | STSBenchmark | 43.04 | 74.06 |
| english | SprintDuplicateQuestions | 35.01 | 91.30 |
| english | StackExchangeClustering.v2 | 42.68 | 51.65 |
| english | StackExchangeClusteringP2P.v2 | 32.63 | 38.55 |
| english | SummEvalSummarization.v2 | 31.72 | 22.58 |
| english | TRECCOVID | 13.31 | 69.23 |
| english | Touche2020Retrieval.v3 | 1.82 | 61.12 |
| english | ToxicConversationsClassification | 60.77 | 62.18 |
| english | TweetSentimentExtractionClassification | 45.08 | 51.07 |
| english | TwentyNewsgroupsClustering.v2 | 14.86 | 33.18 |
| english | TwitterSemEval2015 | 46.17 | 60.53 |
| english | TwitterURLCorpus | 62.04 | 83.16 |
| multilingual | AILAStatutes | 9.95 | 14.91 |
| multilingual | AfriSentiClassification | 36.43 | 38.99 |
| multilingual | AlloProfClusteringS2S.v2 | 17.22 | 39.55 |
| multilingual | AlloprofReranking | 27.56 | 62.56 |
| multilingual | AmazonCounterfactualClassification | 65.69 | 57.25 |
| multilingual | ArXivHierarchicalClusteringP2P | 53.23 | 55.39 |
| multilingual | ArXivHierarchicalClusteringS2S | 50.33 | 51.64 |
| multilingual | ArguAna | 17.04 | 45.12 |
| multilingual | ArmenianParaphrasePC | 91.28 | 93.07 |
| multilingual | BUCC.v2 | 22.29 | 88.49 |
| multilingual | BelebeleRetrieval | 5.92 | 44.05 |
| multilingual | BibleNLPBitextMining | 0.65 | 5.38 |
| multilingual | BigPatentClustering.v2 | 21.60 | 30.47 |
| multilingual | BiorxivClusteringP2P.v2 | 29.97 | 32.91 |
| multilingual | BornholmBitextMining | 5.35 | 25.38 |
| multilingual | BrazilianToxicTweetsClassification | 20.05 | 17.70 |
| multilingual | BulgarianStoreReviewSentimentClassfication | 34.45 | 36.70 |
| multilingual | CEDRClassification | 31.74 | 34.45 |
| multilingual | CLSClusteringP2P.v2 | 27.62 | 32.71 |
| multilingual | CSFDSKMovieReviewSentimentClassification | 20.32 | 22.49 |
| multilingual | CTKFactsNLI | 71.60 | 70.40 |
| multilingual | CataloniaTweetClassification | 45.92 | 45.66 |
| multilingual | Core17InstructionRetrieval | -1.04 | -1.09 |
| multilingual | CovidRetrieval | 9.22 | 54.32 |
| multilingual | CyrillicTurkicLangClassification | 60.14 | 58.64 |
| multilingual | CzechProductReviewSentimentClassification | 43.85 | 45.49 |
| multilingual | DBpediaClassification | 68.93 | 84.13 |
| multilingual | DalajClassification | 50.36 | 49.85 |
| multilingual | DiaBlaBitextMining | 10.53 | 54.04 |
| multilingual | EstonianValenceClassification | 29.62 | 32.04 |
| multilingual | FaroeseSTS | 48.22 | 64.74 |
| multilingual | FilipinoShopeeReviewsClassification | 24.32 | 26.35 |
| multilingual | FinParaSTS | -5.45 | 18.15 |
| multilingual | FinancialPhrasebankClassification | 61.78 | 62.84 |
| multilingual | FloresBitextMining | 1.84 | 27.61 |
| multilingual | GermanSTSBenchmark | 52.98 | 70.23 |
| multilingual | GreekLegalCodeClassification | 2.22 | 11.24 |
| multilingual | GujaratiNewsClassification | 42.99 | 60.16 |
| multilingual | HALClusteringS2S.v2 | 10.17 | 21.17 |
| multilingual | HagridRetrieval | 35.29 | 98.77 |
| multilingual | IN22GenBitextMining | 1.44 | 26.22 |
| multilingual | IndicCrosslingualSTS | 2.26 | 17.15 |
| multilingual | IndicGenBenchFloresBitextMining | 0.73 | 43.70 |
| multilingual | IndicLangClassification | 84.32 | 78.60 |
| multilingual | IndonesianIdClickbaitClassification | 58.07 | 58.46 |
| multilingual | IsiZuluNewsClassification | 21.10 | 23.32 |
| multilingual | ItaCaseholdClassification | 20.63 | 50.50 |
| multilingual | JSICK | 68.47 | 78.80 |
| multilingual | KorHateSpeechMLClassification | 3.23 | 5.93 |
| multilingual | KorSarcasmClassification | 58.76 | 53.21 |
| multilingual | KurdishSentimentClassification | 60.25 | 61.16 |
| multilingual | LEMBPasskeyRetrieval | 2.50 | 25.50 |
| multilingual | LegalBenchCorporateLobbying | 49.63 | 86.86 |
| multilingual | MIRACLRetrievalHardNegatives | 1.82 | 46.37 |
| multilingual | MLQARetrieval | 6.90 | 47.10 |
| multilingual | MacedonianTweetSentimentClassification | 37.03 | 43.13 |
| multilingual | MalteseNewsClassification | 12.25 | 14.38 |
| multilingual | MasakhaNEWSClassification | 49.65 | 58.89 |
| multilingual | MasakhaNEWSClusteringS2S | 29.26 | 37.31 |
| multilingual | MassiveIntentClassification | 25.16 | 47.38 |
| multilingual | MedrxivClusteringP2P.v2 | 29.38 | 33.91 |
| multilingual | MultiEURLEXMultilabelClassification | 1.99 | 2.87 |
| multilingual | MultiHateClassification | 54.59 | 56.71 |
| multilingual | NTREXBitextMining | 5.13 | 48.39 |
| multilingual | NepaliNewsClassification | 66.62 | 85.29 |
| multilingual | News21InstructionRetrieval | 2.29 | -4.14 |
| multilingual | NollySentiBitextMining | 8.44 | 29.36 |
| multilingual | NordicLangClassification | 59.31 | 54.92 |
| multilingual | NorwegianCourtsBitextMining | 72.97 | 90.26 |
| multilingual | NusaParagraphEmotionClassification | 25.96 | 33.45 |
| multilingual | NusaTranslationBitextMining | 26.70 | 54.22 |
| multilingual | NusaX-senti | 58.41 | 61.72 |
| multilingual | NusaXBitextMining | 2.66 | 43.62 |
| multilingual | OdiaNewsClassification | 47.30 | 51.93 |
| multilingual | OpusparcusPC | 86.05 | 87.62 |
| multilingual | PAC | 67.38 | 70.10 |
| multilingual | PawsXPairClassification | 47.79 | 56.42 |
| multilingual | PlscClusteringP2P.v2 | 69.28 | 67.46 |
| multilingual | PoemSentimentClassification | 36.59 | 45.70 |
| multilingual | PolEmo2.0-OUT | 35.20 | 29.64 |
| multilingual | PpcPC | 72.30 | 80.93 |
| multilingual | PunjabiNewsClassification | 47.77 | 65.29 |
| multilingual | RTE3 | 85.01 | 84.82 |
| multilingual | Robust04InstructionRetrieval | -5.88 | -6.97 |
| multilingual | RomaniBibleClustering | 35.74 | 38.53 |
| multilingual | RuBQReranking | 25.53 | 63.30 |
| multilingual | SCIDOCS | 0.72 | 10.59 |
| multilingual | SIB200ClusteringS2S | 5.84 | 13.27 |
| multilingual | SICK-R | 57.94 | 69.47 |
| multilingual | STS12 | 36.61 | 71.03 |
| multilingual | STS13 | 47.17 | 77.23 |
| multilingual | STS14 | 40.18 | 72.83 |
| multilingual | STS15 | 54.63 | 81.90 |
| multilingual | STS17 | 24.21 | 67.11 |
| multilingual | STS22.v2 | 43.82 | 62.55 |
| multilingual | STSB | 57.29 | 72.46 |
| multilingual | STSBenchmark | 43.04 | 74.06 |
| multilingual | STSES | 62.49 | 75.11 |
| multilingual | ScalaClassification | 54.36 | 50.30 |
| multilingual | SemRel24STS | 42.47 | 58.36 |
| multilingual | SentimentAnalysisHindi | 45.05 | 48.88 |
| multilingual | SinhalaNewsClassification | 35.43 | 50.17 |
| multilingual | SiswatiNewsClassification | 33.88 | 57.75 |
| multilingual | SlovakMovieReviewSentimentClassification | 51.14 | 57.16 |
| multilingual | SpartQA | 0.11 | 7.49 |
| multilingual | SprintDuplicateQuestions | 35.01 | 91.30 |
| multilingual | StackExchangeClustering.v2 | 42.68 | 51.65 |
| multilingual | StackOverflowQA | 17.92 | 65.53 |
| multilingual | StatcanDialogueDatasetRetrieval | 0.11 | 16.92 |
| multilingual | SwahiliNewsClassification | 50.42 | 51.40 |
| multilingual | SwednClusteringP2P | 31.11 | 24.30 |
| multilingual | SwissJudgementClassification | 50.54 | 53.55 |
| multilingual | T2Reranking | 54.78 | 67.00 |
| multilingual | TERRa | 54.71 | 52.88 |
| multilingual | TRECCOVID | 13.31 | 69.23 |
| multilingual | Tatoeba | 5.88 | 35.27 |
| multilingual | TempReasonL1 | 0.30 | 0.12 |
| multilingual | ToxicConversationsClassification | 60.77 | 62.18 |
| multilingual | TswanaNewsClassification | 19.38 | 21.95 |
| multilingual | TweetTopicSingleClassification | 41.31 | 59.75 |
| multilingual | TwitterHjerneRetrieval | 8.98 | 36.52 |
| multilingual | TwitterURLCorpus | 62.04 | 83.16 |
| multilingual | VoyageMMarcoReranking | 18.86 | 54.94 |
| multilingual | WebLINXCandidatesReranking | 5.97 | 8.14 |
| multilingual | WikiCitiesClustering | 50.84 | 52.83 |
| multilingual | WikiClusteringP2P.v2 | 19.05 | 24.25 |
| multilingual | WikipediaRerankingMultilingual | 58.29 | 78.00 |
| multilingual | WikipediaRetrievalMultilingual | 18.47 | 70.43 |
| multilingual | WinoGrande | 0.06 | 58.07 |
| multilingual | XNLI | 57.34 | 60.56 |
| multilingual | indonli | 47.41 | 48.72 |
| coir | AppsRetrieval | 0.21 | 5.00 |
| coir | COIRCodeSearchNetRetrieval | 3.31 | 71.47 |
| coir | CodeFeedbackMT | 13.03 | 34.99 |
| coir | CodeFeedbackST | 7.33 | 57.92 |
| coir | CodeSearchNetCCRetrieval | 12.63 | 42.70 |
| coir | CodeTransOceanContest | 9.58 | 45.38 |
| coir | CodeTransOceanDL | 32.05 | 29.87 |
| coir | CosQA | 0.99 | 28.69 |
| coir | StackOverflowQA | 17.92 | 65.53 |
| coir | SyntheticText2SQL | 2.11 | 52.99 |

### MIRACL hard-negative language scores

| Language subset | GOAT base | Previous published |
|---|---:|---:|
| ar | 1.62 | 58.10 |
| bn | 1.60 | 45.44 |
| de | 2.12 | 41.62 |
| en | 0.76 | 48.12 |
| es | 1.07 | 42.59 |
| fa | 2.74 | 39.71 |
| fi | 0.61 | 51.16 |
| fr | 0.97 | 44.06 |
| hi | 4.01 | 32.80 |
| id | 0.64 | 40.80 |
| ja | 0.57 | 52.62 |
| ko | 6.82 | 45.93 |
| ru | 1.00 | 49.19 |
| sw | 2.43 | 44.86 |
| te | 0.28 | 45.93 |
| th | 3.63 | 48.19 |
| yo | 1.47 | 57.31 |
| zh | 0.33 | 46.24 |
