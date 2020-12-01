siesta: siesta_ortho siesta_ice siesta_hm_lda siesta_pbe_nn
	
siesta_ortho:
	echo "SIESTA: Testing atomic (neuralxc) model on water"
	@cd siesta/; sh run_ortho.sh; cd ..

siesta_hm_lda:
	echo "SIESTA: Testing hidden messages LDA model on water"
	@cd siesta/; sh run_ortho_hm_lda.sh; cd ..
	
siesta_pbe_nn:
	echo "SIESTA: Testing neural network PBE on water"
	@cd siesta/; sh run_ortho_pbe_nn.sh; cd ..

siesta_ice:
	echo "SIESTA: Testing atomic water model on ice"
	@cd siesta/; sh run_ice.sh; cd ..



