MSc THESIS DEVELOPMENT PLAN: VFL-RAG DETECTION, EXPLAINABILITY, RAG, RERANKING

1. LIMITATIONS OF THE CURRENT IMPLEMENTATION 
1.1 Sample size and statistics. Attack-class slices in stored results use very few scored examples per class. Any claim that one rerank mode beats another needs confidence intervals and paired tests across the same flows or scenarios, not only row means in a spreadsheet.
1.2 Reference text for metrics. The pipeline can build a deterministic reference string from fixed JSON files that list allowed actions per label and which network tier may execute them. That keeps runs reproducible but penalizes valid wording that is not in the template and can inflate agreement-style scores when the model stays inside a narrow schema.
1.3 Definition of recall. Recall@10 is measured against the stage-one shortlist, typically the top hundred chunks, not against a full corpus labelled for relevance. The number therefore describes how much relevant mass sits in the final ten given the shortlist, not absolute retrieval recall over all chunks.
1.4 Reranker checkpoints. Stage-two models are standard MS MARCO-style checkpoints: MiniLM cross-encoder, ColBERT v2, doc2query T5. None were trained on the cybersecurity knowledge base. Similar scores across modes are therefore plausible and do not by themselves prove that reranking does not matter in security settings.
1.5 doc2query versus BM25. Some aggregate outputs show doc2query and plain BM25 with identical columns. That usually indicates the expanded index is not wired into scoring, or both paths read the same corpus. This should be verified in code before doc2query appears in any thesis table as a distinct method.
1.6 Hybrid retrieval. Dense FAISS and BM25 paths exist separately. Reciprocal rank fusion or another hybrid first stage is not implemented yet. Hybrid baselines are common in heterogeneous retrieval benchmarks; see BEIR: https://arxiv.org/abs/2104.08663
1.7 Cost and latency. Wall-clock time, GPU memory, and throughput per query are not logged in the artifact bundle. Reranking is the main cost knob for a SOC-facing assistant, so the thesis should add at least one cost column next to quality columns.
1.8 Adversarial and poisoning evaluation. Security-focused RAG work in 2026 increasingly reports corpus poisoning and small perturbation attacks. Semantic Chameleon: https://arxiv.org/abs/2603.18034. P3A reranker-focused: https://openreview.net/forum?id=hY720JSxOG. Confundo: https://arxiv.org/abs/2602.06616. RefineRAG: https://arxiv.org/abs/2604.07403. The current stack does not yet run a controlled attack suite or a defence baseline such as ProGRank: https://arxiv.org/abs/2603.22934
1.9 Federated threat model. Vertical federated training is implemented for detection, but label inference, sign-flip, or Byzantine clients are not part of the evaluation story. VFL privacy work such as VMask: https://arxiv.org/abs/2507.14629 sits in the bibliography as context if the thesis adds a security-of-learning angle.
1.10 Single main detection dataset. Results are tied to the CICIDS-2017 slice distributed with the UM-NIDS packaging on IEEE DataPort: https://ieee-dataport.org/documents/unified-multimodal-network-intrusion-detection-systems-dataset A second IDS dataset would support generalisation claims.
1.11 Metric saturation from JSON constraints. Outputs follow a strict JSON schema and an allowed action list. That reduces unsafe verbs but can push automatic truthfulness proxies toward ceiling values. A second optional generation channel without the strict schema, scored separately, can expose subtle grounding failures.
1.12 Query construction. Retrieval queries are template-built from class, confidence, SHAP narrative, and actions. There is no ablation that removes SHAP, or replaces templates with a learned rewriter, so the thesis cannot yet say how much each piece contributes.
1.13 Class-conditioned routing. The idea of choosing the rerank strategy from detector outputs and attribution features is noted in writing but not implemented as a router model.

2. RESEARCH GAPS 
Gap A. Cybersecurity-specific reranker training. 
SciRerankBench covers scientific RAG reranking stress: https://arxiv.org/abs/2508.08742 
SecureBERT-style work targets classification and NER: https://arxiv.org/abs/2510.00240 
A reproducible cross-encoder or ColBERT-style checkpoint trained on NIST, CIS, ISO, and MITRE-style triples and evaluated in the same IDS-to-RAG loop is still a thin part of the literature. 
Downstream-aligned reranker optimisation with LLM feedback is active: https://arxiv.org/abs/2604.02091
Gap B. Poisoning under security-grounded corpora. 
Attacker technique text and defender control text live in adjacent semantic space. Controlled distractor packs plus per-reranker drop metrics are still uncommon compared to open-domain RAG attack papers already cited in 1.8.
Gap C. Federated misbehaviour to retrieval. Most VFL attack papers stop at accuracy or label privacy. This stack feeds SHAP-derived text into queries, so a documented chain from perturbed local embeddings to wrong citations is a systems contribution if measured honestly.
Gap D. Hybrid first stage before reranking. BEIR: https://arxiv.org/abs/2104.08663 motivates hybrid fusion. Published cyber-RAG plus IDS reranker comparisons rarely lead with RRF-style fusion as a headline baseline.
Gap E. Downstream action metrics versus nDCG. CRAG argues for richer RAG benchmark design: https://arxiv.org/abs/2406.04744 Salemi and Zamani show weak correlation between classical relevance and downstream usefulness. SIGIR: https://dl.acm.org/doi/10.1145/3626772.3657957. arXiv: https://arxiv.org/abs/2404.13781. Mitigation-plan scores for action overlap, tier correctness, and citation checks remain rare.
Gap F. CTI LLM benchmarks are not reranker benchmarks. CTIArena: https://openreview.net/forum?id=E0AekM35XF and https://arxiv.org/abs/2510.11974 tests LLMs on heterogeneous cyber threat intelligence. It supports external knowledge value but does not replace reranker-centric experiments.
Gap G. Class-conditioned reranking. Empirical tables already hint attack-dependent behaviour. A router from logits plus SHAP features to mode selection is implementable and underexplored in public repos.
Gap H. LLM judges for security JSON. RAVEN uses task-specific judges on vulnerability reports: https://arxiv.org/abs/2604.17948 Risks of LLM-as-judge are surveyed here: https://arxiv.org/abs/2603.29403 A short rubric on JSON plans is feasible.
Optional. Hierarchical ATT&CK-aware RAG: https://arxiv.org/abs/2604.14166 relates to taxonomy-aware retrieval when framing related work.

3. CHANGES

3.1 Statistical tests on rerank comparisons.
Present state: Score.ipynb loads one rerank_comparison_sample_*.json file per scored flow. The last run used nine files, roughly one per attack class. For each file it computes Recall@10, nDCG@10, SBERT-style grounding scores, CRAG-style proxies, and BERTScore on the LLM overall_reasoning field. Results are aggregated with simple groupby means per rerank mode: dense MMR, BM25, doc2query, dense+crossencoder, BM25+crossencoder, dense+ColBERT. There is no test of whether a 0.02 difference in BERTScore could arise from noise across only nine independent samples.
After this change: The same per-sample metric vectors would be kept, but each mode pair would get a paired test such as Wilcoxon signed-rank on the same flows scored under both modes. Bootstrap confidence intervals on the mean would accompany every table. The thesis could then state that ColBERT beat dense MMR on BERTScore with an interval that excludes zero, instead of relying on a single mean row in a spreadsheet.

3.2 Larger stratified evaluation set.
Present state: Plan.ipynb runs the full six-mode pipeline: retrieve top 100, rerank or truncate to top 10, call GPT-4o-mini once per mode for each selected test flow. It writes rerank_comparison_sample_<id>_<timestamp>.json under RAG_docs/action_plans/. The scoring notebook then averages across however many files exist. A small set makes per-class tables unstable: one odd PORTSCAN case can dominate the class row.
After this change: Hundreds of flows would be drawn from the test split with stratification by predicted label across BENIGN, DDOS, DOS, and the other consolidated classes. Each flow would still produce one JSON bundle with all six modes, but aggregate tables would reflect real variance. Per-class conclusions such as CrossEncoder hurting PORTSCAN would rest on tens of examples per class, not one or two.

3.3 Human gold labels and an LLM judge.
Present state: The reference for almost all metrics is not written by a person. build_reference_text_for_label in utils/rerank_plan_scoring.py reads attack_options.json for which actions belong to each label and agentic_features.json for which tier RAN, Edge, or Core may execute each action. It emits a fixed template string such as PORTSCAN response: RAN actions: block IP, limit rate. The LLM is scored against that template. A plan that picks the right actions with different wording can score low on ROUGE yet be operationally fine. There is no rubric that checks threat level, tier assignment, or whether cited NIST text matches the retrieved chunk titles.
After this change: A subset of fifty to one hundred flows would carry human-written gold rationales or gold action plus tier bundles. A separate LLM judge would score JSON plans on a frozen checklist: actions legal, tiers consistent with SHAP, citations plausible. Correlation between judge scores, human scores, and automatic metrics would show whether nDCG improvements actually track analyst usefulness. That directly supports RQ3.

3.4 Recall measured against human relevance, not only the top-100 pool.
Present state: In Score.ipynb, graded relevance for each chunk is 0.5 times ROUGE-1 plus 0.5 times cosine similarity to the label-derived reference string, with an adaptive threshold from the batch. Recall@10 counts how much of the relevant mass in the candidate_pool, the saved top 100 for that mode, appears in the final top 10. If stage one never surfaces the right NIST control in those hundred slots, recall cannot punish that failure; it only measures reranking inside a possibly already wrong pool.
After this change: For a fixed list of queries built from real detection outputs, annotators would mark chunks in the full corpus or a larger random sample as relevant, partially relevant, or irrelevant. Recall@k would be recomputed against that label set. The thesis could separate two failure modes: bad first-stage recall with nothing good in the hundred versus bad reranking with good chunks in the hundred but the wrong ten shown to the LLM. That clarifies whether to invest in hybrid retrieval, tuning, or reranker training.

3.5 Latency, memory, and throughput per rerank mode.
Present state: Plan.ipynb times are implicit in notebook wall clock only. The saved JSON reports store chunk text and scores but not how long FAISS search, BM25, CrossEncoder batching, ColBERT via RAGatouille, or six OpenAI calls took. An examiner cannot see that dense+ColBERT might be ten times slower than dense MMR for a marginal BERTScore gain.
After this change: Each mode run would log seconds per stage, peak GPU memory if CUDA is used, and batch size. Tables would pair quality columns with cost columns. The discussion chapter could argue for a Pareto choice between ColBERT when latency budget allows and MMR when not, which is standard in retrieval papers but missing today.

3.6 Fix doc2query so it is a real third lexical baseline.
Present state: Plan.ipynb can build an expanded corpus with castorini/doc2query-t5-base-msmarco and cache it in RAG_docs/rag_parents_doc2query.json. The mode doct5query is supposed to run BM25 on that expanded text. When aggregate scores match plain BM25 exactly, the usual cause is that scoring still indexes the non-expanded child texts, or both modes point at the same token list. Until that is traced in Plan.ipynb and the BM25 builder, doc2query is not a distinct experimental condition.
After this change: doc2query would retrieve different top-100 lists than BM25 when expansion changes term overlap. Thesis tables could honestly compare three lexical variants: raw BM25, doc2query-expanded BM25, and dense. If expansion still does not help security text, that becomes a result rather than a wiring bug.

3.7 Hybrid stage-one retrieval before reranking.
Present state: Dense paths use FAISS over MiniLM embeddings with multi-query merge and optional MMR. Lexical paths use BM25 over the full child corpus or the expanded corpus after item 3.6 is fixed. They are separate branches; nothing merges their ranked lists before CrossEncoder or ColBERT sees candidates.
After this change: Reciprocal rank fusion or a simple learned fusion would combine dense and BM25 rankings into one top-100 shortlist, then the same rerankers as today would run unchanged. If hybrid lifts Recall@10 without new training, the thesis gains a cheap baseline. If hybrid plus domain-tuned reranker wins jointly, the story is stronger than either alone. BEIR motivates this pattern: https://arxiv.org/abs/2104.08663

3.8 Query ablations for SHAP and template text.
Present state: Retrieval queries are built from the predicted attack label, confidence, SHAP tier contributions, top feature names, and allowed actions from attack_options.json. That string is what FAISS and BM25 see. There is no experiment that drops SHAP and uses only a class-level mitigation query, or that replaces the template with an LLM-rewritten query. DSPy: https://github.com/stanfordnlp/dspy
After this change: Four query variants would run through the same index: class name only; class plus single strongest SHAP feature; full current template; optional LLM rewrite of the full template. Differences in Recall@10 and downstream JSON quality would show whether attribution-conditioned retrieval is worth the complexity, which supports the architecture chapter and limits over-claiming SHAP value.

3.9 Unconstrained rationale channel for evaluation only.
Present state: prompt_rag.txt forces strict JSON: threat_level, primary_actions with tier and reasoning, knowledge_sources_used, and actions must come from the allowed list. Score.ipynb then compares overall_reasoning to the template reference. CRAG-style LLM scores often sit at 1.0 because the schema prevents blatantly wrong verbs even when the narrative cites the wrong control family or ignores retrieved text.
After this change: A second optional call would ask for a short free-text analyst summary without the action whitelist, scored only on grounding against retrieved chunks and human rubrics. Saturation on constrained metrics would no longer hide hallucinated policy references. The production system could still ship the JSON plan; the thesis would report both channels.

3.10 Reproducible packaging and config files.
Present state: The workflow is Train.ipynb, Predict.ipynb, Index.ipynb, Plan.ipynb, Score.ipynb. Hyperparameters for top 100, top 10, MMR lambda, and model names live in notebook cells. Re-running a figure for the thesis means re-executing long notebooks and hoping cell order and environment match.
After this change: A small package or scripted CLI with one YAML or Hydra config per experiment would record embedding model, reranker checkpoint path, sample IDs, and random seeds. Each thesis table row would map to a config hash in an appendix. Examiners and future readers could reproduce a single row without re-running the entire five-notebook chain.

4. ADDITION

Task 1. Domain-tuned reranker.
Present state: Stage two uses off-the-shelf checkpoints trained on MS MARCO web search pairs: cross-encoder/ms-marco-MiniLM-L-6-v2 for dense and BM25 shortlists, ColBERT v2 via RAGatouille for dense shortlists. Those models learned to rank general web passages, not NIST SP 800-53 control paragraphs versus CIS mapping sentences that share vocabulary but differ in obligation. That is the main reason current tables show similar nDCG and small BERTScore spreads: the rerankers are not specialised.
After this addition: Training triples would be mined from the existing knowledge base: query text similar to Plan.ipynb queries, positive passage equals a relevant parent section, negatives equals random or hard negatives from other controls. A fine-tuned cross-encoder and optionally ColBERT would replace the MS MARCO weights in the same Plan.ipynb loop. The thesis contribution becomes a released checkpoint plus evidence that domain tuning separates rankings on policy-heavy queries where generic rerankers tie. Methods: https://arxiv.org/abs/1901.04085 and https://arxiv.org/abs/2004.12832

Task 2. Adversarial or stress-test benchmark split.
Present state: All retrieved chunks come from the honest corpus built in Index.ipynb from NIST, CIS, MITRE, and ISO PDFs. There is no protocol that injects misleading chunks and measures how often they reach the top 10 or skew the JSON plan. Security RAG papers in 2026 expect at least one such axis. Semantic Chameleon: https://arxiv.org/abs/2603.18034. P3A: https://openreview.net/forum?id=hY720JSxOG
After this addition: For each test query, controlled packs would be appended or swapped into the candidate pool: unrelated controls as noise, controls from a neighbouring attack class as semantic look-alike, or attacker-flavoured text phrased like guidance as counterfactual. Each rerank mode would be scored on clean versus stressed conditions and on drop in CRAG or action-aware metrics. Optional ProGRank-style filtering: https://arxiv.org/abs/2603.22934 would show whether a defence recovers ranking. The thesis could claim a security-aware reranker evaluation, not only benign-corpus IR scores. SciRerankBench motivates the stress dimensions: https://arxiv.org/abs/2508.08742

Task 3. Malicious vertical-partition simulation.
Present state: Train.ipynb trains three local encoders; only embeddings fuse at the server. Predict.ipynb outputs class, confidence, and SHAP contributions per tier. That SHAP text drives retrieval queries. The training code does not simulate a client that sends biased embeddings or corrupted features at inference, and no experiment measures whether such bias changes which PDF sections appear in the prompt.
After this addition: One party’s input would be perturbed through scaled features, swapped statistics, or a label-flip style attack during training if time allows. The chain would be measured stepwise: classification change, SHAP narrative change, query string change, top-10 chunk IDs change, JSON action change. A defence story could cite VMask-style privacy: https://arxiv.org/abs/2507.14629 or robust aggregation. This answers RQ4 and fills Gap C: federated detection papers rarely connect to RAG outcomes.

Task 4. Class-conditioned reranker router.
Present state: Every flow runs all six retrieval modes; Score.ipynb compares them post hoc. Published tables already show DDOS behaving differently from PORTSCAN, including cases where CrossEncoder collapses CRAG on PORTSCAN. Nothing automatically picks dense+ColBERT for DDOS and dense MMR for PORTSCAN at inference time.
After this addition: A small classifier would take predicted label, confidence vector, and SHAP tier weights and output a mode choice. Training labels would come from which mode won on historical samples, with an oracle upper bound reported separately. A learned router would be compared to always-ColBERT and always-MMR. The thesis gains a practical SOC recommendation plus analysis of when routing helps, tied to Gap G.

Task 5. Action-aware metrics.
Present state: Metrics are mostly retrieval-centric such as Recall@10 and nDCG@10, or similarity of overall_reasoning to the template string via SBERT, ROUGE, BERTScore, and CRAG proxies. They do not directly score whether the JSON primary_actions list matches what a senior analyst would assign to the dominant tier, or whether block IP was assigned to perimeter when SHAP said access-tier dominated.
After this addition: Explicit scores would include fraction of gold actions recovered, false actions invented, tier match rate per action, and whether knowledge_sources_used names align with retrieved titles. These would be correlated with nDCG to test Gap E. Salemi and Zamani: https://arxiv.org/abs/2404.13781. A reranker that lifts nDCG but lowers action F1 would be flagged as misleading, which is a strong thesis figure.

Task 6. Second dataset or corpus for transfer.
Present state: Detection metrics and RAG examples are tied to the undersampled CICIDS-2017 file from UM-NIDS on IEEE DataPort: https://ieee-dataport.org/documents/unified-multimodal-network-intrusion-detection-systems-dataset The knowledge base is policy PDFs, not traffic from another year or vendor.
After this addition: A second IDS CSV such as UNSW-NB15 or a held-out slice would run through Predict.ipynb with the same VFL model or a retrained head, producing new queries. Alternatively only retrieval would be tested on a fresh advisory corpus. The thesis could include one transfer table: tuned reranker trained on NIST-heavy triples, tested on queries from the second dataset. That supports generalisation claims beyond a single 2017 traffic dump.

7. READING LIST 
Methods: 
SciRerankBench https://arxiv.org/abs/2508.08742, 
BEIR https://arxiv.org/abs/2104.08663, 
CRAG https://arxiv.org/abs/2406.04744, 
ColBERT https://arxiv.org/abs/2004.12832, 
passage reranking with BERT https://arxiv.org/abs/1901.04085
Alignment and evaluation: 
LLM-feedback reranking https://arxiv.org/abs/2604.02091, 
Salemi and Zamani SIGIR https://dl.acm.org/doi/10.1145/3626772.3657957 and arXiv https://arxiv.org/abs/2404.13781
Attacks and defences: 
Semantic Chameleon https://arxiv.org/abs/2603.18034, P3A https://openreview.net/forum?id=hY720JSxOG, 
Confundo https://arxiv.org/abs/2602.06616, 
RefineRAG https://arxiv.org/abs/2604.07403, 
ProGRank https://arxiv.org/abs/2603.22934
Cybersecurity context: 
SecureBERT 2.0 https://arxiv.org/abs/2510.00240, 
ReGAIN https://arxiv.org/abs/2512.22223 and code https://github.com/270771/llm-traffic, 
FeRAG listing https://researchr.org/publication/FuHMKSD26, 
CyberGuardian 2 https://doi.org/10.5220/0013406000003928 , federated RAG mapping https://arxiv.org/abs/2505.18906
Tooling: 
RAGAS https://github.com/explodinggradients/ragas, 
original RAG https://arxiv.org/abs/2005.11401, 
SHAP https://www.jmlr.org/papers/volume22/20-1317/20-1317.pdf, 
ReasonerRank https://aclanthology.org/2025.findings-acl.107/

8. RESEARCH QUESTIONS
RQ1. Does fine-tuning a reranker on cybersecurity policy and threat-intel text change ranking behaviour and downstream mitigation metrics versus off-the-shelf MS MARCO-style rerankers in the same two-stage pipeline.
RQ2. Under controlled poisoning or distractor injection, which reranker architectures degrade least, and does a training-free defence change the ordering. ProGRank: https://arxiv.org/abs/2603.22934
RQ3. Do action-aware or LLM-judge scores rank rerankers differently from nDCG and embedding similarity, and how do they correlate with a small human-labelled slice.
RQ4. Do perturbed vertical-partition embeddings or explanations shift queries and citations enough to matter, and what mitigations reduce drift. VMask: https://arxiv.org/abs/2507.14629

9. THESIS CHAPTER PLAN
Ch.1 Introduction. About six to eight pages and near two thousand words. 
Contents: operational motivation for distributed detection and grounded response, problem statement, research questions listed in Section 8, bullet contributions, thesis roadmap.
Ch.2 Background. About twelve to fifteen pages and near four thousand words. 
Contents: intrusion detection features and labels, vertical federated learning definition and why embeddings cross a boundary, SHAP at a high level, large language model limits, RAG pipeline stages, bi-encoder versus cross-encoder versus late interaction, MMR intuition, MITRE and NIST and CIS and ISO roles as citation sources.

Ch.3 Related work and gap positioning. About ten to twelve pages and near thirty-five hundred words. 
Contents: three threads on federated or privacy-aware detection, SOC assistants and agentic LLM tooling, and retrieval and reranking benchmarks. End with a written matrix in prose: prior systems versus privacy, explainability, reranker tuning, robustness, downstream task metrics. Point to gaps in Section 2.

Ch.4 System architecture. About eight to ten pages and near twenty-eight hundred words. 
Contents: end-to-end Detect to Reason to Act diagram, data flow from partitioned features to logits to SHAP text to queries to FAISS and BM25 to rerankers to JSON schema, file and notebook touchpoints.

Ch.5 Core contribution domain-tuned reranker. About ten to twelve pages and near thirty-five hundred words. 
Contents: triple construction, negatives mining, training schedule, ablations, tables versus baselines.

Ch.6 Core contribution adversarial or stress benchmark. About ten to twelve pages and near thirty-five hundred words. 
Contents: threat model, injection protocol, metrics, optional defence run.

Ch.7 Core contribution evaluation upgrade. About eight to ten pages and near twenty-eight hundred words.
Contents: action-aware metric definitions, judge prompts, human study protocol and agreement stats.

Ch.8 Experiments. About fourteen to eighteen pages and near five thousand words. 
Contents: datasets and hardware, full quantitative tables, per-class discussion, statistics from 3.1, latency from 3.5, two or three qualitative walkthroughs.

Ch.9 Discussion. About six to eight pages and near two thousand words. 
Contents: validity threats, dual-use and ethics of publishing attack recipes, deployment limits.

Ch.10 Conclusion. About four to six pages and near fifteen hundred words. 
Contents: answers mapped to RQ1 through RQ4, explicit limits, three short future lines.

Back matter. References often add eight to twelve printed pages. Appendices for prompts, annotation guidelines, long tables, and config hashes often add ten to twenty pages. Front matter including abstract under about three hundred words often adds near ten pages. Total bound volume often lands near one hundred ten to one hundred thirty pages for a UK or EU style MSc if appendices stay lean.

10. FOURTEEN-WEEK SCHEDULE
Week 1. Finish Tier 1 reading from Section 7. Write a one-page gap table for the supervisor and lock RQ wording.
Week 2. Implement statistical tests and larger stratified sample draws. Start doc2query path audit from item 3.6. Begin experiment config logging with YAML or Hydra.
Week 3. Build triple dataset for reranker training. Train first domain-tuned cross-encoder checkpoint. Smoke test on held-out queries.
Week 4. Complete Task 1 ablations and draft Chapter 5 body text with figures.
Week 5. Design Task 2 stress splits and implement injection hooks in the planning notebook path.
Week 6. Run Task 2 experiments and optional ProGRank baseline. Draft Chapter 6.
Week 7. Implement Task 5 metrics and frozen LLM judge rubric. Pilot fifty-case human labels.
Week 8. Expand human labels toward target hundred or switch to deep case studies if labels stall.
Week 9. Freeze configs. Full sweep for all figures and tables intended for Chapter 8. Collect latency numbers.
Week 10. Assemble first full draft of Chapters 1 through 8 and send to supervisor.
Week 11. Integrate supervisor feedback. Draft Discussion and Conclusion chapters.
Week 12. Second revision pass. Tighten abstract, introduction, and contribution bullets to match final results.
Week 13. Appendices, reference hygiene, print layout check, figure resolution check.
Week 14. Final formatting, PDF for submission, oral defence preparation if applicable.

11. THESIS-SIZED GAP
Scholar searches combining cybersecurity, cross-encoder, reranker, and fine-tune still return few public repositories that run intrusion detection outputs into a policy-grounded RAG loop with released reranker checkpoints. Gap A in Section 2 therefore remains plausible provided evaluation is strengthened as in Sections 3 and 4.
