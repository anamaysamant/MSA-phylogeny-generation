import pickle as pkl

FAMILIES = ["PF00271","PF00005","PF00004","PF01535","PF13354","PF00595","PF00397","PF00153","PF07679",
            "PF00076","PF00072","PF00096","PF00512","PF00041","PF02518"]
            # "PF01356","PF03440","PF04008","PF06351","PF06355", "PF16747","PF18648"]
# FAMILIES = ["PF00004"]
MSA_TYPES = ["seed"]
CONTEXT_TYPES = ["static"]
CONTEXT_SIZES = ["10"]
CONTEXT_SAMPLINGS = ["random"]
PROPOSAL_TYPES = ["logits"]
N_MUTATIONS = ["500"]
N_SEQUENCES = ["50"]
INIT_SEQS = ["0"]

N_MUTATIONS_START = 500
N_MUTATIONS_END = ["1000"]

num_simulations = 1

SIM_INDS = list(range(1,num_simulations+1))
SIM_INDS = list(map(str,SIM_INDS))

MUTATION_INTERVALS = ["20"]
N_SEQUENCES_MI = ["100"]
START_SEQS = ["sampled"]
N_ROUNDS = ["100"]
PSEUDOCOUNTS = ["0.0","0.3","0.4"]

R_EFFS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0]

with open("families_under_200.pkl","rb") as f:

    FAMILIES = pkl.load(f)

rule all:
    input:
        # expand("scores/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.tsv",
        #         msa_type = MSA_TYPES, context_sampling = CONTEXT_SAMPLINGS, context_size = CONTEXT_SIZES, 
        #         fam = FAMILIES, sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        # expand("scores/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.tsv",
        #         msa_type = MSA_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, 
        #         sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        # expand("data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.fasta",
        #         msa_type = MSA_TYPES, context_sampling = CONTEXT_SAMPLINGS, context_size = CONTEXT_SIZES, 
        #         fam = FAMILIES, sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        expand("data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.fasta",
                msa_type = MSA_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, 
                sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        # expand("data/msa-{msa_type}-simulation-trees/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.newick",
        #         msa_type = MSA_TYPES, context_sampling = CONTEXT_SAMPLINGS, context_size = CONTEXT_SIZES, 
        #         fam = FAMILIES, sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        # expand("data/msa-{msa_type}-simulation-trees/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.newick",
        #         msa_type = MSA_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, 
        #         sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        # expand("scores/msa-{msa_type}-sim-trees/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-tree.tsv",
        #         msa_type = MSA_TYPES, context_sampling = CONTEXT_SAMPLINGS, context_size = CONTEXT_SIZES, 
        #         fam = FAMILIES, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        # expand("scores/msa-{msa_type}-sim-trees/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-tree.tsv",
        #         msa_type = MSA_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, 
        #         proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        # expand("scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.tsv",
        #         msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, n_mutations = N_MUTATIONS_END, 
        #         context_size = CONTEXT_SIZES, n_sequences = N_SEQUENCES, init_seq = INIT_SEQS),
        # expand("data/MI_decay_MSAs/{mutation_interval}-mutation-interval/{n_sequences_MI}-sequences/{n_rounds}-rounds/msa-{msa_type}-simulations/MSA-1b/init-seq-{start_seqs}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.tsv",
        #         mutation_interval=MUTATION_INTERVALS, n_sequences_MI = N_SEQUENCES_MI,msa_type = MSA_TYPES, start_seqs = START_SEQS, 
        #         proposal_type = PROPOSAL_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, sim_ind = SIM_INDS,n_rounds = N_ROUNDS),
        # expand("data/MI_decay_MSAs/{mutation_interval}-mutation-interval/{n_sequences_MI}-sequences/{n_rounds}-rounds/msa-{msa_type}-simulations/MSA-1b/init-seq-{start_seqs}/{proposal_type}-proposal/context-size-{context_size}/{fam}/MI-decay-MI-values/pseudo-{pseudocount}/{fam}-{sim_ind}-MI-pseudo-{pseudocount}.tsv",
        #         mutation_interval=MUTATION_INTERVALS, n_sequences_MI = N_SEQUENCES_MI,msa_type = MSA_TYPES, start_seqs = START_SEQS, 
        #         proposal_type = PROPOSAL_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, sim_ind = SIM_INDS,n_rounds = N_ROUNDS,
        #         pseudocount = PSEUDOCOUNTS),
        # expand("figures/MI-decay-analysis/{mutation_interval}-mutation-interval/{n_sequences_MI}-sequences/{n_rounds}-rounds/msa-{msa_type}-simulations/MSA-1b/init-seq-{start_seqs}/context-size-{context_size}/{fam}/{fam}-MI-pseudo-{pseudocount}.jpg",
        #         mutation_interval=MUTATION_INTERVALS, n_sequences_MI = N_SEQUENCES_MI,msa_type = MSA_TYPES, start_seqs = START_SEQS, 
        #         proposal_type = PROPOSAL_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, sim_ind = SIM_INDS,n_rounds = N_ROUNDS,
        #         pseudocount = PSEUDOCOUNTS),
        # expand("other-analyses/r-effective-analysis/scores/{fam}/init-seq-{init_seq}/scaling-{r_eff}/static-context/{context_size}/{fam}-{sim_ind}.tsv",
        #         msa_type = MSA_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, 
        #         sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS, r_eff = R_EFFS),
        # expand("other-analyses/r-effective-analysis/data/{fam}/init-seq-{init_seq}/scaling-{r_eff}/static-context/{context_size}/{fam}-{sim_ind}.fasta",
        #         msa_type = MSA_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, 
        #         sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS, r_eff = R_EFFS),


rule generate_tree_MSA_static:
    input:
        "data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.fasta"
    output:
        "data/msa-{msa_type}-simulation-trees/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.newick"
    log:
        "logs/msa-{msa_type}-simulation-trees/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.log"
    shell:
        "fasttree {input} > {output}"

rule generate_tree_MSA_dynamic:
    input:
        "data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.fasta"
    output:
        "data/msa-{msa_type}-simulation-trees/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.newick"
    log:
        "logs/msa-{msa_type}-simulation-trees/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.log"
    shell:
        "fasttree {input} > {output}"

rule generate_tree_metrics_static:
    input:
        simulated_trees=expand("data/msa-{msa_type}-simulation-trees/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.newick",
                sim_ind = SIM_INDS, allow_missing = True), 
        simulated_MSAs=expand("data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.fasta",
                sim_ind = SIM_INDS, allow_missing = True),
        seed_MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        seed_tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"     
    output:
        "scores/msa-{msa_type}-sim-trees/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-tree.tsv"
    script:
        "scripts/tree_metrics_generator.py"

rule generate_tree_metrics_dynamic:
    input:
        simulated_trees=expand("data/msa-{msa_type}-simulation-trees/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.newick",
                sim_ind = SIM_INDS, allow_missing = True), 
        simulated_MSAs=expand("data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.fasta",
                sim_ind = SIM_INDS, allow_missing = True),
        seed_MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta", 
        seed_tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"     
    output:
        "scores/msa-{msa_type}-sim-trees/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-tree.tsv"
    script:
        "scripts/tree_metrics_generator.py"

rule root_tree:
    input:
        "data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/{msa_type}-trees/{fam}_{msa_type}_rooted.newick"
    shell:
        "python scripts/root_tree.py --input {input} --output {output}"

rule convert_fasta_to_a2m_MSA_static:
    input:
        hmm="data/protein-families-hmms/{fam}.hmm",
        simulated_MSA="data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.fasta",
    output:
        a2m_file="data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-a2m/{fam}-{sim_ind}.a2m",
        ungapped_seq=temp("data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}-ungapped.fasta")
    shell:
        """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmalign -o {output.a2m_file} --outformat a2m {input.hmm} {output.ungapped_seq}
        """

rule convert_fasta_to_a2m_MSA_dynamic:
    input:
        hmm="data/protein-families-hmms/{fam}.hmm",
        simulated_MSA="data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.fasta",
    output:
        a2m_file="data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-a2m/{fam}-{sim_ind}.a2m",
        ungapped_seq=temp("data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}-ungapped.fasta")
    shell:
        """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmalign -o {output.a2m_file} --outformat a2m {input.hmm} {output.ungapped_seq}
        """

# rule simulate_without_phylogeny_MSA:
#     input:
#         MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
#     output:
#         "data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.fasta"
#     log:
#         "logs/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.log"
#     shell:
#         """
#         python scripts/simulate_along_phylogeny.py --tool MSA_1b --output {output} --input_MSA {input.MSA} --context_size {wildcards.context_size}  \
#         --proposal_type {wildcards.proposal_type} --seed {wildcards.sim_ind} --no_phylogeny --n_mutations {wildcards.n_mutations} \
#          --n_sequences {wildcards.n_sequences} --start_seq_index {wildcards.init_seq}
#         """

rule simulate_without_phylogeny_extended_MSA:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        simulated_MSA=f"data/no-phylogeny/{N_MUTATIONS_START}-mutations/{{n_sequences}}-sequences/msa-{{msa_type}}-simulations/MSA-1b/init-seq-{{init_seq}}/{{proposal_type}}-proposal/context-size-{{context_size}}/{{fam}}/{{fam}}-{{sim_ind}}.fasta",
    output:
        "data/no-phylogeny/{n_mutations_end}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.fasta"
    resources:
        nvidia_gpu=1
    params:
        n_mutations_start= N_MUTATIONS_START
    log:
        "logs/no-phylogeny/{n_mutations_end}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.log"
    shell:
        """
        python scripts/extend_MSA_simulation.py --tool MSA_1b --output {output} --input_MSA {input.MSA} --simulated_MSA {input.simulated_MSA} --context_size {wildcards.context_size}  \
        --proposal_type {wildcards.proposal_type} --seed {wildcards.sim_ind} --n_mutations_start {params.n_mutations_start} --n_mutations_end {wildcards.n_mutations_end} \
        """


rule simulate_for_MI_decorrelation_MSA:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
    output:
        "data/MI_decay_MSAs/{mutation_interval}-mutation-interval/{n_sequences_MI}-sequences/{n_rounds}-rounds/msa-{msa_type}-simulations/MSA-1b/init-seq-{start_seqs}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.tsv"
    resources:
        nvidia_gpu=1
    log:
        "logs/MI_decay_MSAs/{mutation_interval}-mutation-interval/{n_sequences_MI}-sequences/{n_rounds}-rounds/msa-{msa_type}-simulations/MSA-1b/init-seq-{start_seqs}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.log"
    shell:
        """
        python scripts/gen_MI_decay_MSA.py  --output {output} --input_MSA {input.MSA} --context_size {wildcards.context_size}  \
        --proposal_type {wildcards.proposal_type} --seed {wildcards.sim_ind} --n_mutations_interval {wildcards.mutation_interval} \
         --n_sequences {wildcards.n_sequences_MI} --n_rounds {wildcards.n_rounds} --start_seqs {wildcards.start_seqs}
        """

rule calculate_MI_decorrelation_MSA:
    input:
        "data/MI_decay_MSAs/{mutation_interval}-mutation-interval/{n_sequences_MI}-sequences/{n_rounds}-rounds/msa-{msa_type}-simulations/MSA-1b/init-seq-{start_seqs}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.tsv"
    output:
        "data/MI_decay_MSAs/{mutation_interval}-mutation-interval/{n_sequences_MI}-sequences/{n_rounds}-rounds/msa-{msa_type}-simulations/MSA-1b/init-seq-{start_seqs}/{proposal_type}-proposal/context-size-{context_size}/{fam}/MI-decay-MI-values/pseudo-{pseudocount}/{fam}-{sim_ind}-MI-pseudo-{pseudocount}.tsv"
    shell:
        """
        python scripts/MI_decay.py  --output {output} --input_MSA {input}  \
        --proposal_type {wildcards.proposal_type} --pseudocount {wildcards.pseudocount} --sim_ind {wildcards.sim_ind}  \
        """

rule plot_MI_decorrelation_MSA:
    input:
        MI_files=expand("data/MI_decay_MSAs/{mutation_interval}-mutation-interval/{n_sequences_MI}-sequences/{n_rounds}-rounds/msa-{msa_type}-simulations/MSA-1b/init-seq-{start_seqs}/{proposal_type}-proposal/context-size-{context_size}/{fam}/MI-decay-MI-values/pseudo-{pseudocount}/{fam}-{sim_ind}-MI-pseudo-{pseudocount}.tsv",
                proposal_type = PROPOSAL_TYPES,sim_ind = SIM_INDS, allow_missing = True),
        seed_MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta"       
    output:
        "figures/MI-decay-analysis/{mutation_interval}-mutation-interval/{n_sequences_MI}-sequences/{n_rounds}-rounds/msa-{msa_type}-simulations/MSA-1b/init-seq-{start_seqs}/context-size-{context_size}/{fam}/{fam}-MI-pseudo-{pseudocount}.jpg"
    log:
        "logs/MI-decay-analysis/{mutation_interval}-mutation-interval/{n_sequences_MI}-sequences/{n_rounds}-rounds/msa-{msa_type}-simulations/MSA-1b/init-seq-{start_seqs}/context-size-{context_size}/{fam}/{fam}-MI-pseudo-{pseudocount}.log"
    script:
        "scripts/plot_MI_decay_correlation.py"


rule simulate_along_phylogeny_MSA_static:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.fasta"
    resources:
        nvidia_gpu=1
    log:
        "logs/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.log"
    shell:
        """
        python scripts/simulate_along_phylogeny.py --tool MSA_1b --output {output} --input_MSA {input.MSA} --input_tree {input.tree} \
        --context_type static --context_size {wildcards.context_size} --proposal_type {wildcards.proposal_type} --seed {wildcards.sim_ind}\
        --start_seq_index {wildcards.init_seq} --log_file {log}
        """

rule simulate_along_phylogeny_MSA_dynamic:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.fasta"
    resources:
        nvidia_gpu=1
    log:
        "logs/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.log"
    shell:
        """
        python scripts/simulate_along_phylogeny.py --tool MSA_1b --output {output} --input_MSA {input.MSA} --input_tree {input.tree} \
        --context_type dynamic --context_size {wildcards.context_size} --context_sampling {wildcards.context_sampling} \
        --proposal_type {wildcards.proposal_type} --seed {wildcards.sim_ind} --start_seq_index {wildcards.init_seq} --log_file {log}
        """

rule generate_scores_MSA_static:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy",
        pdb_path = "data/pdb-ref-structures/{fam}-ref.pdb"
    output:
        ungapped_seq=temp("data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.tbl"),
        score_table="scores/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-{sim_ind}.tsv"
    shell:
        """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --max --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree} --pdb_path {input.pdb_path}
        """


rule generate_scores_MSA_dynamic:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy",
        pdb_path = "data/pdb-ref-structures/{fam}-ref.pdb"
    output:
        ungapped_seq=temp("data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.tbl"),
        score_table="scores/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-{sim_ind}.tsv"
    shell:
        """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --max --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree} --pdb_path {input.pdb_path}
        """

rule generate_scores_no_phylogeny_MSA:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy",
        pdb_path = "data/pdb-ref-structures/{fam}-ref.pdb"

    output:
        ungapped_seq=temp("data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.tbl"),
        score_table="scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.tsv"
    shell:
       """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --max --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree} --no_phylogeny --pdb_path {input.pdb_path}
        """


rule simulate_along_phylogeny_MSA_static_r_eff:
    input:
        MSA="data/protein-families-msa-seed/{fam}_seed.fasta",
        tree="data/seed-trees/{fam}_seed.newick"
    output:
        "other-analyses/r-effective-analysis/data/{fam}/init-seq-{init_seq}/scaling-{r_eff}/static-context/{context_size}/{fam}-{sim_ind}.fasta"
    resources:
        nvidia_gpu=1
    shell:
        """
        python scripts/simulate_along_phylogeny.py --tool MSA_1b --output {output} --input_MSA {input.MSA} --input_tree {input.tree} \
        --context_type static --context_size {wildcards.context_size} --proposal_type logits --seed {wildcards.sim_ind}\
        --start_seq_index {wildcards.init_seq} --r_eff {wildcards.r_eff}
        """

rule generate_scores_MSA_static_r_eff:
    input:
        original_MSA_seed="data/protein-families-msa-seed/{fam}_seed.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="other-analyses/r-effective-analysis/data/{fam}/init-seq-{init_seq}/scaling-{r_eff}/static-context/{context_size}/{fam}-{sim_ind}.fasta",
        tree="data/seed-trees/{fam}_seed.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy",
        pdb_path = "data/pdb-ref-structures/{fam}-ref.pdb"
    output:
        ungapped_seq=temp("other-analyses/r-effective-analysis/data/{fam}/init-seq-{init_seq}/scaling-{r_eff}/static-context/{context_size}/{/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("other-analyses/r-effective-analysis/scores/{fam}/init-seq-{init_seq}/scaling-{r_eff}/static-context/{context_size}/{fam}-{sim_ind}.tbl"),
        score_table="other-analyses/r-effective-analysis/scores/{fam}/init-seq-{init_seq}/scaling-{r_eff}/static-context/{context_size}/{fam}-{sim_ind}.tsv"
    shell:
        """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --max --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree} --pdb_path {input.pdb_path}
        """