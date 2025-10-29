from llm_term_extractor.llm_term_generator_1_1 import run_llm_term_generation
from nld_generator.nld_generator_1_4 import run_nld_generation
from term_categorizer.term_categorizer_1_5 import run_term_categorization

def main():
    run_llm_term_generation()
    run_nld_generation()
    run_term_categorization()

if __name__ == "__main__":
    main()
