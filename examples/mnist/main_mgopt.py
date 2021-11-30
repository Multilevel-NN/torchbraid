#@HEADER
# ************************************************************************
# 
#                        Torchbraid v. 0.1
# 
# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC 
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. 
# Government retains certain rights in this software.
# 
# Torchbraid is licensed under 3-clause BSD terms of use:
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name National Technology & Engineering Solutions of Sandia, 
# LLC nor the names of the contributors may be used to endorse or promote 
# products derived from this software without specific prior written permission.
# 
# Questions? Contact Eric C. Cyr (eccyr@sandia.gov)
# 
# ************************************************************************
#@HEADER

###
# Examples to compare TB+NI vs. TB+MG/Opt vs. TB+MG/Opt+Local
###
#
#  +++ TB+NI (no multilevel MGRIT) +++
# $ python3 main_mgopt.py --steps 8 --samp-ratio 0.1 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2  --ni-levels 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --mgopt-iter 0
#    ....
#    ....
#    Train Epoch: 2 [0/5000 (0%)]        	Loss: 0.080145	Time Per Batch 0.074357
#    Train Epoch: 2 [500/5000 (10%)]     	Loss: 0.166830	Time Per Batch 0.073399
#    Train Epoch: 2 [1000/5000 (20%)]     	Loss: 0.291933	Time Per Batch 0.074495
#    Train Epoch: 2 [1500/5000 (30%)]     	Loss: 0.273469	Time Per Batch 0.074411
#    Train Epoch: 2 [2000/5000 (40%)]     	Loss: 0.237901	Time Per Batch 0.074258
#    Train Epoch: 2 [2500/5000 (50%)]     	Loss: 0.044225	Time Per Batch 0.075304
#    Train Epoch: 2 [3000/5000 (60%)]     	Loss: 0.450144	Time Per Batch 0.075509
#    Train Epoch: 2 [3500/5000 (70%)]     	Loss: 0.221938	Time Per Batch 0.075396
#    Train Epoch: 2 [4000/5000 (80%)]     	Loss: 0.170495	Time Per Batch 0.075297
#    Train Epoch: 2 [4500/5000 (90%)]     	Loss: 0.054780	Time Per Batch 0.075103
#    Train Epoch: 2 [5000/5000 (100%)]     	Loss: 0.529004	Time Per Batch 0.074992
#  
#    Test set: Average loss: 0.0107, Accuracy: 859/1000 (86%)
#
#
# +++ TB+MG/Opt (Takes the above NI solver and adds 1 epoch of MG/Opt)  (no multilevel MGRIT) +++
# $ python3 main_mgopt.py --steps 12 --samp-ratio 0.1 --epochs 2 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1
#    ....
#    ....
#    Train Epoch: 1 [50/5000 (1%)]     	Loss: 0.023389	Time Per Batch 0.780689
#    Train Epoch: 1 [550/5000 (11%)]     	Loss: 0.037635	Time Per Batch 0.781935
#    Train Epoch: 1 [1050/5000 (21%)]     	Loss: 0.171758	Time Per Batch 0.781161
#    Train Epoch: 1 [1550/5000 (31%)]     	Loss: 0.355388	Time Per Batch 0.781739
#    Train Epoch: 1 [2050/5000 (41%)]     	Loss: 0.286502	Time Per Batch 0.785055
#    Train Epoch: 1 [2550/5000 (51%)]     	Loss: 0.007334	Time Per Batch 0.787544
#    Train Epoch: 1 [3050/5000 (61%)]     	Loss: 0.472483	Time Per Batch 0.788244
#    Train Epoch: 1 [3550/5000 (71%)]     	Loss: 0.497987	Time Per Batch 0.788385
#    Train Epoch: 1 [4050/5000 (81%)]     	Loss: 0.550990	Time Per Batch 0.789357
#    Train Epoch: 1 [4550/5000 (91%)]     	Loss: 0.502712	Time Per Batch 0.794309
#    Train Epoch: 1 [5000/5000 (100%)]     	Loss: 1.374733	Time Per Batch 0.795855
#
#     Test set: Average loss: 0.0350, Accuracy: 830/1000 (83%)
#
#     Time per epoch: 8.04e+01 
#     Time per test:  8.51e-01 
#     Train Epoch: 2 [50/5000 (1%)]     	Loss: 0.735889	Time Per Batch 0.807446
#     Train Epoch: 2 [550/5000 (11%)]     	Loss: 0.490954	Time Per Batch 0.980087
#     Train Epoch: 2 [1050/5000 (21%)]     	Loss: 0.328517	Time Per Batch 0.899957
#     Train Epoch: 2 [1550/5000 (31%)]     	Loss: 0.291075	Time Per Batch 0.889309
#     Train Epoch: 2 [2050/5000 (41%)]     	Loss: 0.590438	Time Per Batch 0.907808
#     Train Epoch: 2 [2550/5000 (51%)]     	Loss: 0.002660	Time Per Batch 0.921411
#     Train Epoch: 2 [3050/5000 (61%)]     	Loss: 0.377914	Time Per Batch 0.933670
#     Train Epoch: 2 [3550/5000 (71%)]     	Loss: 0.170308	Time Per Batch 0.921680
#     Train Epoch: 2 [4050/5000 (81%)]     	Loss: 0.247056	Time Per Batch 0.947788
#     Train Epoch: 2 [4550/5000 (91%)]     	Loss: 0.700092	Time Per Batch 0.992239
#     Train Epoch: 2 [5000/5000 (100%)]     	Loss: 0.660585	Time Per Batch 1.025534
#
#     Test set: Average loss: 0.0316, Accuracy: 874/1000 (87%)
#
#
# ++++  TB+MG/Opt+Local (Takes the above NI solver and adds 2 epoch of MG/Opt with purely local relaxation on each level) +++
# $ python3 main_mgopt.py --steps 8 --samp-ratio 0.1 --epochs 2 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-fwd-finefcf --lp-bwd-finefcf --lp-fwd-relaxonlycg --lp-bwd-relaxonlycg --lp-fwd-finalrelax --lp-iters 1
#  pretty bad results
#
#
#
#   No Forward relax only
# $ python3 main_mgopt.py --steps 12 --samp-ratio 0.1 --epochs 4 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-bwd-finefcf --lp-bwd-relaxonlycg --lp-iters 1
#    ....
#    ....
#   Train Epoch: 1 [50/5000 (1%)]     	Loss: 0.026627	Time Per Batch 1.354365
#   Train Epoch: 1 [550/5000 (11%)]     	Loss: 0.038312	Time Per Batch 1.314453
#   Train Epoch: 1 [1050/5000 (21%)]     	Loss: 0.145884	Time Per Batch 1.302440
#   Train Epoch: 1 [1550/5000 (31%)]     	Loss: 0.409986	Time Per Batch 1.304109
#   Train Epoch: 1 [2050/5000 (41%)]     	Loss: 0.249901	Time Per Batch 1.301143
#   Train Epoch: 1 [2550/5000 (51%)]     	Loss: 0.015579	Time Per Batch 1.303860
#   Train Epoch: 1 [3050/5000 (61%)]     	Loss: 0.432999	Time Per Batch 1.303941
#   Train Epoch: 1 [3550/5000 (71%)]     	Loss: 0.537103	Time Per Batch 1.304572
#   Train Epoch: 1 [4050/5000 (81%)]     	Loss: 0.509826	Time Per Batch 1.309897
#   Train Epoch: 1 [4550/5000 (91%)]     	Loss: 0.425591	Time Per Batch 1.319231
#   Train Epoch: 1 [5000/5000 (100%)]     	Loss: 1.695210	Time Per Batch 1.327237

#     Test set: Average loss: 0.0339, Accuracy: 856/1000 (86%)

#     Time per epoch: 1.33e+02 
#     Time per test:  8.37e-01 
#   Train Epoch: 2 [50/5000 (1%)]     	Loss: 0.757971	Time Per Batch 1.322812
#   Train Epoch: 2 [550/5000 (11%)]     	Loss: 0.387885	Time Per Batch 1.627624
#   Train Epoch: 2 [1050/5000 (21%)]     	Loss: 0.484076	Time Per Batch 1.511711
#   Train Epoch: 2 [1550/5000 (31%)]     	Loss: 0.063839	Time Per Batch 1.460629
#   Train Epoch: 2 [2050/5000 (41%)]     	Loss: 0.459545	Time Per Batch 1.465456
#   Train Epoch: 2 [2550/5000 (51%)]     	Loss: 0.083642	Time Per Batch 1.487204
#   Train Epoch: 2 [3050/5000 (61%)]     	Loss: 0.232119	Time Per Batch 1.508128
#   Train Epoch: 2 [3550/5000 (71%)]     	Loss: 0.866607	Time Per Batch 1.483077
#   Train Epoch: 2 [4050/5000 (81%)]     	Loss: 0.251402	Time Per Batch 1.510811
#   Train Epoch: 2 [4550/5000 (91%)]     	Loss: 0.249597	Time Per Batch 1.524319
#   Train Epoch: 2 [5000/5000 (100%)]     	Loss: 0.944854	Time Per Batch 1.559652

#     Test set: Average loss: 0.0370, Accuracy: 859/1000 (86%)

#     Time per epoch: 1.45e+02 (1 std dev 1.64e+01)
#     Time per test:  8.39e-01 (1 std dev 2.74e-03)
#   Train Epoch: 3 [50/5000 (1%)]     	Loss: 0.131977	Time Per Batch 1.661334
#   Train Epoch: 3 [550/5000 (11%)]     	Loss: 0.104281	Time Per Batch 1.951927
#   Train Epoch: 3 [1050/5000 (21%)]     	Loss: 0.473015	Time Per Batch 1.720850
#   Train Epoch: 3 [1550/5000 (31%)]     	Loss: 0.488855	Time Per Batch 1.706517
#   Train Epoch: 3 [2050/5000 (41%)]     	Loss: 0.370757	Time Per Batch 1.683737
#   Train Epoch: 3 [2550/5000 (51%)]     	Loss: 0.045061	Time Per Batch 1.725872
#   Train Epoch: 3 [3050/5000 (61%)]     	Loss: 0.020187	Time Per Batch 1.806705
#   Train Epoch: 3 [3550/5000 (71%)]     	Loss: 0.006413	Time Per Batch 1.788941
#   Train Epoch: 3 [4050/5000 (81%)]     	Loss: 0.239261	Time Per Batch 1.844042
#   Train Epoch: 3 [4550/5000 (91%)]     	Loss: 0.026802	Time Per Batch 1.899658
#   Train Epoch: 3 [5000/5000 (100%)]     	Loss: 0.414071	Time Per Batch 1.935275

#     Test set: Average loss: 0.0330, Accuracy: 868/1000 (87%)

#     Time per epoch: 1.61e+02 (1 std dev 3.07e+01)
#     Time per test:  8.37e-01 (1 std dev 4.04e-03)
#   Train Epoch: 4 [50/5000 (1%)]     	Loss: 0.061009	Time Per Batch 1.923087
#   Train Epoch: 4 [550/5000 (11%)]     	Loss: 0.016384	Time Per Batch 2.153438
#   Train Epoch: 4 [1050/5000 (21%)]     	Loss: 0.017974	Time Per Batch 1.884625
#   Train Epoch: 4 [1550/5000 (31%)]     	Loss: 0.061684	Time Per Batch 1.995264
#   Train Epoch: 4 [2050/5000 (41%)]     	Loss: 0.735824	Time Per Batch 2.060233
#   Train Epoch: 4 [2550/5000 (51%)]     	Loss: 0.025946	Time Per Batch 2.074943
#   Train Epoch: 4 [3050/5000 (61%)]     	Loss: 0.136023	Time Per Batch 2.172852
#   Train Epoch: 4 [3550/5000 (71%)]     	Loss: 0.003095	Time Per Batch 2.141511
#   Train Epoch: 4 [4050/5000 (81%)]     	Loss: 0.015791	Time Per Batch 2.147388
#   Train Epoch: 4 [4550/5000 (91%)]     	Loss: 0.006470	Time Per Batch 2.200998
#   Train Epoch: 4 [5000/5000 (100%)]     	Loss: 0.361463	Time Per Batch 2.216012

#     Test set: Average loss: 0.0349, Accuracy: 880/1000 (88%)

#     Time per epoch: 1.77e+02 (1 std dev 3.94e+01)
#     Time per test:  8.37e-01 (1 std dev 3.55e-03)
#
#
#
#
# Same as above, but turned off relaxonlycg 
# $ python3 main_mgopt.py --steps 12 --samp-ratio 0.1 --epochs 4 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1  --lp-iters 1
#    ....
#    ....
#   Train Epoch: 1 [50/5000 (1%)]     	Loss: 0.023389	Time Per Batch 0.791520
#   Train Epoch: 1 [550/5000 (11%)]     	Loss: 0.037635	Time Per Batch 0.786753
#   Train Epoch: 1 [1050/5000 (21%)]     	Loss: 0.171758	Time Per Batch 0.787739
#   Train Epoch: 1 [1550/5000 (31%)]     	Loss: 0.355388	Time Per Batch 0.787124
#   Train Epoch: 1 [2050/5000 (41%)]     	Loss: 0.286502	Time Per Batch 0.792335
#   Train Epoch: 1 [2550/5000 (51%)]     	Loss: 0.007334	Time Per Batch 0.792362
#   Train Epoch: 1 [3050/5000 (61%)]     	Loss: 0.472483	Time Per Batch 0.793232
#   Train Epoch: 1 [3550/5000 (71%)]     	Loss: 0.497987	Time Per Batch 0.793867
#   Train Epoch: 1 [4050/5000 (81%)]     	Loss: 0.550990	Time Per Batch 0.794216
#   Train Epoch: 1 [4550/5000 (91%)]     	Loss: 0.502712	Time Per Batch 0.799245
#   Train Epoch: 1 [5000/5000 (100%)]     	Loss: 1.374733	Time Per Batch 0.801235

#     Test set: Average loss: 0.0350, Accuracy: 830/1000 (83%)

#     Time per epoch: 8.09e+01 
#     Time per test:  8.45e-01 
#   Train Epoch: 2 [50/5000 (1%)]     	Loss: 0.735889	Time Per Batch 0.807867
#   Train Epoch: 2 [550/5000 (11%)]     	Loss: 0.490954	Time Per Batch 0.985419
#   Train Epoch: 2 [1050/5000 (21%)]     	Loss: 0.328517	Time Per Batch 0.904941
#   Train Epoch: 2 [1550/5000 (31%)]     	Loss: 0.291075	Time Per Batch 0.888753
#   Train Epoch: 2 [2050/5000 (41%)]     	Loss: 0.590438	Time Per Batch 0.903429
#   Train Epoch: 2 [2550/5000 (51%)]     	Loss: 0.002660	Time Per Batch 0.911302
#   Train Epoch: 2 [3050/5000 (61%)]     	Loss: 0.377914	Time Per Batch 0.921025
#   Train Epoch: 2 [3550/5000 (71%)]     	Loss: 0.170308	Time Per Batch 0.907940
#   Train Epoch: 2 [4050/5000 (81%)]     	Loss: 0.247056	Time Per Batch 0.905952
#   Train Epoch: 2 [4550/5000 (91%)]     	Loss: 0.700092	Time Per Batch 0.927277
#   Train Epoch: 2 [5000/5000 (100%)]     	Loss: 0.660585	Time Per Batch 0.938034

#     Test set: Average loss: 0.0316, Accuracy: 874/1000 (87%)

#     Time per epoch: 8.77e+01 (1 std dev 9.67e+00)
#     Time per test:  8.85e-01 (1 std dev 5.58e-02)
#   Train Epoch: 3 [50/5000 (1%)]     	Loss: 0.062921	Time Per Batch 1.369416
#   Train Epoch: 3 [550/5000 (11%)]     	Loss: 0.036601	Time Per Batch 1.013770
#   Train Epoch: 3 [1050/5000 (21%)]     	Loss: 0.039995	Time Per Batch 0.995541
#   Train Epoch: 3 [1550/5000 (31%)]     	Loss: 0.009170	Time Per Batch 0.987893
#   Train Epoch: 3 [2050/5000 (41%)]     	Loss: 0.183021	Time Per Batch 0.999170
#   Train Epoch: 3 [2550/5000 (51%)]     	Loss: 0.196579	Time Per Batch 1.025507
#   Train Epoch: 3 [3050/5000 (61%)]     	Loss: 0.030743	Time Per Batch 1.043898
#   Train Epoch: 3 [3550/5000 (71%)]     	Loss: 0.018446	Time Per Batch 1.031856
#   Train Epoch: 3 [4050/5000 (81%)]     	Loss: 0.075926	Time Per Batch 1.069943
#   Train Epoch: 3 [4550/5000 (91%)]     	Loss: 0.017131	Time Per Batch 1.096766
#   Train Epoch: 3 [5000/5000 (100%)]     	Loss: 0.507160	Time Per Batch 1.150955

#     Test set: Average loss: 0.0334, Accuracy: 881/1000 (88%)

#     Time per epoch: 9.71e+01 (1 std dev 1.76e+01)
#     Time per test:  8.69e-01 (1 std dev 4.81e-02)
#   Train Epoch: 4 [50/5000 (1%)]     	Loss: 0.056383	Time Per Batch 1.205629
#   Train Epoch: 4 [550/5000 (11%)]     	Loss: 0.007187	Time Per Batch 1.092589
#   Train Epoch: 4 [1050/5000 (21%)]     	Loss: 0.026061	Time Per Batch 1.111170
#   Train Epoch: 4 [1550/5000 (31%)]     	Loss: 0.003648	Time Per Batch 1.100019
#   Train Epoch: 4 [2050/5000 (41%)]     	Loss: 0.144006	Time Per Batch 1.152981
#   Train Epoch: 4 [2550/5000 (51%)]     	Loss: 0.012323	Time Per Batch 1.189198
#   Train Epoch: 4 [3050/5000 (61%)]     	Loss: 0.016588	Time Per Batch 1.242436
#   Train Epoch: 4 [3550/5000 (71%)]     	Loss: 0.171187	Time Per Batch 1.247144
#   Train Epoch: 4 [4050/5000 (81%)]     	Loss: 0.000800	Time Per Batch 1.278146
#   Train Epoch: 4 [4550/5000 (91%)]     	Loss: 0.005176	Time Per Batch 1.303955
#   Train Epoch: 4 [5000/5000 (100%)]     	Loss: 0.069220	Time Per Batch 1.322476

#     Test set: Average loss: 0.0275, Accuracy: 902/1000 (90%)

#     Time per epoch: 1.06e+02 (1 std dev 2.30e+01)
#     Time per test:  8.63e-01 (1 std dev 4.09e-02)
#
#
#
# +++ relax_onlycg used both ways +++
#  $python3 main_mgopt.py --steps 12 --samp-ratio 0.1 --epochs 4 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-bwd-finefcf --lp-bwd-relaxonlycg --lp-iters 1 --lp-fwd-relaxonlycg --lp-fwd-finefcf 
#
#   Train Epoch: 1 [50/5000 (1%)]     	Loss: 1.007567	Time Per Batch 1.445430
#   Train Epoch: 1 [550/5000 (11%)]     	Loss: 1.922850	Time Per Batch 1.413595
#   Train Epoch: 1 [1050/5000 (21%)]     	Loss: 2.457091	Time Per Batch 1.406766
#   Train Epoch: 1 [1550/5000 (31%)]     	Loss: 2.268729	Time Per Batch 1.403638
#   Train Epoch: 1 [2050/5000 (41%)]     	Loss: 0.977772	Time Per Batch 1.401830
#   Train Epoch: 1 [2550/5000 (51%)]     	Loss: 0.631171	Time Per Batch 1.401039
#   Train Epoch: 1 [3050/5000 (61%)]     	Loss: 1.675838	Time Per Batch 1.400502
#   Train Epoch: 1 [3550/5000 (71%)]     	Loss: 1.321176	Time Per Batch 1.399944
#   Train Epoch: 1 [4050/5000 (81%)]     	Loss: 1.194556	Time Per Batch 1.401059
#   Train Epoch: 1 [4550/5000 (91%)]     	Loss: 0.882402	Time Per Batch 1.402261
#   Train Epoch: 1 [5000/5000 (100%)]     	Loss: 2.066770	Time Per Batch 1.402217

#     Test set: Average loss: 0.2011, Accuracy: 97/1000 (10%)

#     Time per epoch: 1.41e+02 
#     Time per test:  1.19e+00 
#   Train Epoch: 2 [50/5000 (1%)]     	Loss: 1.722834	Time Per Batch 1.459755
#   Train Epoch: 2 [550/5000 (11%)]     	Loss: 1.460086	Time Per Batch 1.401814
#   Train Epoch: 2 [1050/5000 (21%)]     	Loss: 1.757288	Time Per Batch 1.402223
#   Train Epoch: 2 [1550/5000 (31%)]     	Loss: 2.199386	Time Per Batch 1.403022
#   Train Epoch: 2 [2050/5000 (41%)]     	Loss: 1.552675	Time Per Batch 1.401290
#   Train Epoch: 2 [2550/5000 (51%)]     	Loss: 0.758706	Time Per Batch 1.401679
#   Train Epoch: 2 [3050/5000 (61%)]     	Loss: 1.310882	Time Per Batch 1.401103
#   Train Epoch: 2 [3550/5000 (71%)]     	Loss: 0.891020	Time Per Batch 1.400442
#   Train Epoch: 2 [4050/5000 (81%)]     	Loss: 0.944256	Time Per Batch 1.400451
#   Train Epoch: 2 [4550/5000 (91%)]     	Loss: 1.046933	Time Per Batch 1.399794
#   Train Epoch: 2 [5000/5000 (100%)]     	Loss: 1.556541	Time Per Batch 1.401413

#     Test set: Average loss: 0.2360, Accuracy: 104/1000 (10%)

#     Time per epoch: 1.41e+02 (1 std dev 5.64e-02)
#     Time per test:  1.19e+00 (1 std dev 2.97e-03)
#   Train Epoch: 3 [50/5000 (1%)]     	Loss: 2.034826	Time Per Batch 1.398959
#   Train Epoch: 3 [550/5000 (11%)]     	Loss: 0.837276	Time Per Batch 1.407873
#   Train Epoch: 3 [1050/5000 (21%)]     	Loss: 1.266515	Time Per Batch 1.402009
#   Train Epoch: 3 [1550/5000 (31%)]     	Loss: 2.104429	Time Per Batch 1.401355
#   Train Epoch: 3 [2050/5000 (41%)]     	Loss: 1.407837	Time Per Batch 1.400941
#   Train Epoch: 3 [2550/5000 (51%)]     	Loss: 0.856600	Time Per Batch 1.402929
#   Train Epoch: 3 [3050/5000 (61%)]     	Loss: 0.871899	Time Per Batch 1.401853
#   Train Epoch: 3 [3550/5000 (71%)]     	Loss: 1.080811	Time Per Batch 1.401933
#   Train Epoch: 3 [4050/5000 (81%)]     	Loss: 0.813038	Time Per Batch 1.400869
#   Train Epoch: 3 [4550/5000 (91%)]     	Loss: 0.583654	Time Per Batch 1.400475
#   Train Epoch: 3 [5000/5000 (100%)]     	Loss: 1.266221	Time Per Batch 1.402157

#     Test set: Average loss: 0.2344, Accuracy: 111/1000 (11%)

#     Time per epoch: 1.41e+02 (1 std dev 4.28e-02)
#     Time per test:  1.19e+00 (1 std dev 1.10e-02)
#   Train Epoch: 4 [50/5000 (1%)]     	Loss: 1.248186	Time Per Batch 1.414158
#   Train Epoch: 4 [550/5000 (11%)]     	Loss: 0.844222	Time Per Batch 1.400704
#   Train Epoch: 4 [1050/5000 (21%)]     	Loss: 1.108292	Time Per Batch 1.399791
#   Train Epoch: 4 [1550/5000 (31%)]     	Loss: 2.273903	Time Per Batch 1.401236
#   Train Epoch: 4 [2050/5000 (41%)]     	Loss: 1.012678	Time Per Batch 1.399921
#   Train Epoch: 4 [2550/5000 (51%)]     	Loss: 0.578300	Time Per Batch 1.399089
#   Train Epoch: 4 [3050/5000 (61%)]     	Loss: 1.157738	Time Per Batch 1.399163
#   Train Epoch: 4 [3550/5000 (71%)]     	Loss: 0.872913	Time Per Batch 1.398733
#   Train Epoch: 4 [4050/5000 (81%)]     	Loss: 1.091227	Time Per Batch 1.417358
#   Train Epoch: 4 [4550/5000 (91%)]     	Loss: 1.097214	Time Per Batch 1.434556
#   Train Epoch: 4 [5000/5000 (100%)]     	Loss: 2.205729	Time Per Batch 1.437537

#     Test set: Average loss: 0.3024, Accuracy: 107/1000 (11%)
#
#

from __future__ import print_function
import numpy as np

import torch
from torchvision import datasets, transforms
from mpi4py import MPI
from mgopt import parse_args, mgopt_solver

def main():
  
  ##
  # Parse command line args (function defined above)
  args = parse_args()
  procs = MPI.COMM_WORLD.Get_size()
  rank  = MPI.COMM_WORLD.Get_rank()
  
  ##
  # Load training and testing data, while reducing the number of samples (if desired) for faster execution
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                                 ])
  dataset = datasets.MNIST('./digit-data', download=False,transform=transform)
  train_size = int(50000*args.samp_ratio)
  test_size = int(10000*args.samp_ratio)
  #
  train_set = torch.utils.data.Subset(dataset,range(train_size))
  test_set  = torch.utils.data.Subset(dataset,range(train_size,train_size+test_size))
  #
  train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False)
  print("\nTraining setup:  Batch size:  " + str(args.batch_size) + "  Sample ratio:  " + str(args.samp_ratio) + "  MG/Opt Epochs:  " + str(args.epochs) )
  
  ##
  # Compute number of nested iteration steps, going from fine to coarse
  ni_steps = np.array([int(args.steps/(args.ni_rfactor**(args.ni_levels-i-1))) for i in range(args.ni_levels)])
  ni_steps = ni_steps[ ni_steps != 0 ]
  local_ni_steps = np.flip( np.array(ni_steps / procs, dtype=int) )
  print("\nNested iteration steps:  " + str(ni_steps))

  ##
  # Define ParNet parameters for each nested iteration level, starting from fine to coarse
  networks = [] 
  for lsteps in local_ni_steps: 
    networks.append( ('ParallelNet', {'channels'          : args.channels, 
                                      'local_steps'       : lsteps,
                                      'max_iters'         : args.lp_iters,
                                      'print_level'       : args.lp_print,
                                      'Tf'                : args.tf,
                                      'max_fwd_levels'    : args.lp_fwd_levels,
                                      'max_bwd_levels'    : args.lp_bwd_levels,
                                      'max_fwd_iters'     : args.lp_fwd_iters,
                                      'print_level'       : args.lp_print,
                                      'braid_print_level' : args.lp_braid_print,
                                      'fwd_cfactor'       : args.lp_fwd_cfactor,
                                      'bwd_cfactor'       : args.lp_bwd_cfactor,
                                      'fine_fwd_fcf'      : args.lp_fwd_finefcf,
                                      'fine_bwd_fcf'      : args.lp_bwd_finefcf,
                                      'fwd_nrelax'        : args.lp_fwd_nrelax_coarse,
                                      'bwd_nrelax'        : args.lp_bwd_nrelax_coarse,
                                      'skip_downcycle'    : not args.lp_use_downcycle,
                                      'fmg'               : args.lp_use_fmg,
                                      'fwd_relax_only_cg' : args.lp_fwd_relaxonlycg,
                                      'bwd_relax_only_cg' : args.lp_bwd_relaxonlycg,
                                      'CWt'               : args.lp_use_crelax_wt,
                                      'fwd_finalrelax'    : args.lp_fwd_finalrelax
                                      }))
                                 
  ##
  # Specify optimization routine on each level, starting from fine to coarse
  #optims = [ ("pytorch_sgd", { 'lr':args.lr, 'momentum':0.9}) for i in range(len(ni_steps)) ]
  optims = [ ("pytorch_adam", { 'lr':0.001, 'betas':(0.9, 0.999), 'eps':1e-08 }) for i in range(len(ni_steps)) ]

  ##
  # Initialize MG/Opt solver with nested iteration 
  epochs = args.NIepochs
  mgopt_printlevel = args.mgopt_printlevel
  log_interval = args.log_interval
  mgopt = mgopt_solver()
  mgopt.initialize_with_nested_iteration(ni_steps, train_loader, test_loader,
          networks, epochs=epochs, log_interval=log_interval,
          mgopt_printlevel=mgopt_printlevel, optims=optims, seed=args.seed) 
   
  print(mgopt)
  mgopt.options_used()
  
  ##
  # Turn on for fixed-point test.  
  # Works when running  $$ python3 main_mgopt.py --samp-ratio 0.002 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --mgopt-printlevel 3 --batch-size 1
  if False:
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    train_set = torch.utils.data.Subset(dataset, [1])
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=False)
    for (data,target) in train_loader:  pass
    model = mgopt.levels[0].model
    with torch.no_grad():
      model.eval()
      output = model(data)
      loss = model.compose(criterion, output, target)
    
    print("Doing fixed point test.  Loss on single training example should be zero: " + str(loss.item()))
    model.train()

 # Can change MGRIT options from NI to MG/Opt with the following
 #mgopt.levels[0].model.parallel_nn.setFwdNumRelax(0,level=0) 
  
  ##
  # Run the MG/Opt solver
  #   Note: that we use the default restrict and interp options, but these can be modified on a per-level basis
  if( args.mgopt_iter > 0):
    epochs = args.epochs
    line_search = ('tb_simple_ls', {'ls_params' : {'alphas' : [0.01, 0.1, 0.5, 1.0, 2.0, 4.0]}} )
    log_interval = args.log_interval
    mgopt_printlevel = args.mgopt_printlevel
    mgopt_iter = args.mgopt_iter
    mgopt_levels = args.mgopt_levels
    mgopt_tol=0
    nrelax_pre = args.mgopt_nrelax_pre
    nrelax_post = args.mgopt_nrelax_post
    nrelax_coarse = args.mgopt_nrelax_coarse
    mgopt.mgopt_solve(train_loader, test_loader, epochs=epochs,
            log_interval=log_interval, mgopt_tol=mgopt_tol,
            mgopt_iter=mgopt_iter, nrelax_pre=nrelax_pre,
            nrelax_post=nrelax_post, nrelax_coarse=nrelax_coarse,
            mgopt_printlevel=mgopt_printlevel, mgopt_levels=mgopt_levels,
            line_search=line_search)
   
    print(mgopt)
    mgopt.options_used()
  ##
  


if __name__ == '__main__':
  main()



