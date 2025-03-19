FAMILIES = ["PF00004"]
MSA_TYPES = ["seed"]
CONTEXT_TYPES = ["static","dynamic"]
CONTEXT_SIZES = ["10"]
CONTEXT_SAMPLINGS = ["greedy","random"]
PROPOSAL_TYPES = ["msa_prob_dist","random"]
N_MUTATIONS = ["500"]
N_SEQUENCES = ["50"]
INIT_SEQS = ["0"]

N_MUTATIONS_START = 500
N_MUTATIONS_END = ["1000"]

num_simulations = 10

SIM_INDS = list(range(1,num_simulations+1))
SIM_INDS = list(map(str,SIM_INDS))

rule all:
    input:
        expand("scores/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}.tsv",
                msa_type = MSA_TYPES, context_sampling = CONTEXT_SAMPLINGS, context_size = CONTEXT_SIZES, 
                fam = FAMILIES, sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        expand("scores/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}.tsv",
                msa_type = MSA_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, 
                sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        # expand("data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-a2m/{fam}-{sim_ind}.a2m",
        #         msa_type = MSA_TYPES, context_sampling = CONTEXT_SAMPLINGS, context_size = CONTEXT_SIZES, 
        #         fam = FAMILIES, sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        # expand("data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-a2m/{fam}-{sim_ind}.a2m",
        #         msa_type = MSA_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, 
        #         sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        # expand("scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.tsv",
        #         msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, n_mutations = N_MUTATIONS_END, 
        #         context_size = CONTEXT_SIZES, n_sequences = N_SEQUENCES, init_seq = INIT_SEQS),
        expand("data/msa-{msa_type}-simulation-trees/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}.newick",
                msa_type = MSA_TYPES, context_sampling = CONTEXT_SAMPLINGS, context_size = CONTEXT_SIZES, 
                fam = FAMILIES, sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),
        expand("data/msa-{msa_type}-simulation-trees/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}.newick",
                msa_type = MSA_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, 
                sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, init_seq = INIT_SEQS),


rule generate_tree_MSA_static:
    input:
        "data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}.fasta"
    output:
        "data/msa-{msa_type}-simulation-trees/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}.newick"
    log:
        "logs/msa-{msa_type}-simulation-trees/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}.log"
    shell:
        "FastTree {input} > {output}"

rule generate_tree_MSA_dynamic:
    input:
        "data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}.fasta"
    output:
        "data/msa-{msa_type}-simulation-trees/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}.newick"
    log:
        "logs/msa-{msa_type}-simulation-trees/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}.log"
    shell:
        "FastTree {input} > {output}"

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
        simulated_MSA="data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}.fasta",
    output:
        a2m_file="data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}-a2m/{fam}-{sim_ind}.a2m",
        ungapped_seq=temp("data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}-ungapped.fasta")
    shell:
        """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmalign -o {output.a2m_file} --outformat a2m {input.hmm} {output.ungapped_seq}
        """

rule convert_fasta_to_a2m_MSA_dynamic:
    input:
        hmm="data/protein-families-hmms/{fam}.hmm",
        simulated_MSA="data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}.fasta",
    output:
        a2m_file="data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}-a2m/{fam}-{sim_ind}.a2m",
        ungapped_seq=temp("data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}-ungapped.fasta")
    shell:
        """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmalign -o {output.a2m_file} --outformat a2m {input.hmm} {output.ungapped_seq}
        """

# rule simulate_without_phylogeny_MSA:
#     input:
#         MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
#     output:
#         "data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.fasta"
#     log:
#         "logs/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.log"
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
        "data/no-phylogeny/{n_mutations_end}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.fasta"
    params:
        n_mutations_start= N_MUTATIONS_START
    log:
        "logs/no-phylogeny/{n_mutations_end}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.log"
    shell:
        """
        python scripts/extend_MSA_simulation.py --tool MSA_1b --output {output} --input_MSA {input.MSA} --simulated_MSA {input.simulated_MSA} --context_size {wildcards.context_size}  \
        --proposal_type {wildcards.proposal_type} --seed {wildcards.sim_ind} --n_mutations_start {params.n_mutations_start} --n_mutations_end {wildcards.n_mutations_end} \
        """


rule simulate_along_phylogeny_MSA_static:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}.fasta"
    log:
        "logs/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}.log"
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
        "data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}.fasta"
    log:
        "logs/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}.log"
    shell:
        """
        python scripts/simulate_along_phylogeny.py --tool MSA_1b --output {output} --input_MSA {input.MSA} --input_tree {input.tree} \
        --context_type dynamic --context_size {wildcards.context_size} --context_sampling {wildcards.context_sampling} \
        --proposal_type {wildcards.proposal_type} --seed {wildcards.sim_ind} --start_seq_index {wildcards.init_seq} --log_file {log}
        """

rule simulate_along_phylogeny_MSA_chunked:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/chunked/{context_size}/{fam}/{fam}-{sim_ind}.fasta"
    shell:
        """
        python scripts/simulate_along_phylogeny.py --output {output} --input_MSA {input.MSA} --input_tree {input.tree} \
        --tool MSA-1b --context_type static --context_size {wildcards.context_size} --chunked --seed {wildcards.sim_ind} \
        --proposal_type {wildcards.proposal_type} --start_seq_index {wildcards.init_seq} --log_file {log}
        """

rule generate_scores_MSA_static:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"
    output:
        ungapped_seq=temp("data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}.tbl"),
        score_table="scores/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/static-context/{context_size}/{fam}/{fam}-{sim_ind}.tsv"
    shell:
        """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --max --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree}
        """


rule generate_scores_MSA_dynamic:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"
    output:
        ungapped_seq=temp("data/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}.tbl"),
        score_table="scores/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/dynamic-context/{context_size}/{context_sampling}/{fam}/{fam}-{sim_ind}.tsv"
    shell:
        """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --max --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree}
        """

rule generate_scores_no_phylogeny_MSA:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"

    output:
        ungapped_seq=temp("data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.tbl"),
        score_table="scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.tsv"
    shell:
       """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --max --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree} --no_phylogeny
        """
