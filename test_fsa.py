#from tests.fsa.test_cmp_attn_decode import test_cmp_attn_decode
#test_cmp_attn_decode()

#from tests.nsa.benchmark_nsa import benchmark
#benchmark.run(print_data=True, save_path='.')

#from tests.flash_mla.test_flash_mla_decoding import main
#main(torch.bfloat16)

#from tests.flash_mla.test_flash_mla_prefill import main
#main()

from tests.txl_mla.test_txl_mla_prefill import main
main()
