

results/clustering_performace_results_formatted.xlsx: main.py means.py
	mkdir -p results
	python main.py


clean:
	rm -r results
	
.PHONY: all clean
