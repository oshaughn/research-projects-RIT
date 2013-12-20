

<< MyGraphics`FigureFormatting`
<< AppliedStatistics`Nonparametric`PointsToDensity`KernelEstimators`
<< AppliedStatistics`Nonparametric`PointsToDensity`WeightedKernelEstimators`
<< PhysicalApplications`Astrophysics`GravitationalWaves`InspiralWaves`

(* LAZY VERSION:
    (m1,m2,filename) list available, stored in 'massToFileNames'
   Each 'filename' is a *-result.dat file
 *)
mc[m1_, m2_] := (m1 m2)^(3/5)/(m1 + m2)^(1/5)
eta[m1_, m2_] := m1 m2/(m1 + m2)^2


mergedResultForPair[mpair_] := 
 Module[{dat, nEvals, maxL, lnZ}, 
  fnames = Select[massToFileNames, (#[[1]] == mpair) &][[All, 2]];
  dat =  Import[#, "Table"][[2, {1,3,4}]] & /@ fnames;
  nEvals = Plus @@ dat[[All,-1]];
  maxL = Max[dat[[All,2]]];
  lnZ = Log[ Plus@@ (  (#[[1]] #[[3]]/nEvals)&/@dat)  ];
  {mpair[[1]], mpair[[2]],  lnZ, maxL, nEvals}
  ]


ROSOneDimensionalWeightedHistogram[bps_BayesianPosteriorSamples, name_, width_, range_, opts___] := Module[{dat, expr,x},
   dat = {#[[1]], Exp[#[[4]]] #[[3]]/#[[2]]}&/@ bps[name, "p", "ps", "lnL"];
   expr = EstimateDensityWithWeightedKernel[x, dat, EstimationKernel1d0, width];
   Plot[expr, {x, range[[1]], range[[2]]}, opts]
]

ROSTwoDimensionalWeightedHistogram[bps_BayesianPosteriorSamples, param1List_, param2List_,  opts___] := Module[{dat, expr,x,y, name1,width1, range1,name2,range2,width2},

  {name1, width1, range1} = param1List;
  {name2, width2, range2} = param2List;
   dat = {{#[[1]],#[[2]]}, Exp[#[[-1]]] #[[-3]]/#[[-2]]}&/@ bps[name1, name2, "p", "ps", "lnL"];
   expr = EstimateDensityWithWeightedKernelAnisotropic[{x,y}, dat, EstimationKernelNd0, {width1, width2}];
   Plot3D[expr, {x, range1[[1]], range1[[2]]},{y, range2[[1]], range2[[2]]}, opts]
]


(* PREFERRED MASS PRIOR *)
pdfM1M2PriorRaw[m1_, m2_]  := 
  If[m1 > 1 && m2 > 1 && m1 + m2 < 30, 1, 0];
pdfM1M2PriorNorm = 
  NIntegrate[pdfM1M2PriorRaw[m1, m2], {m1, 0, 50}, {m2, 0, 50}];
pdfM1M2Prior // Clear;
pdfM1M2Prior = 
 Function @@ {{m1, m2}, 
   pdfM1M2PriorRaw[m1, m2]/
    pdfM1M2PriorNorm} ; (* uniform in m1,m2 from 1 to 30*)

ROSComputeEvidence[m1m2ZList_, pdfM1M2Prior_] := Module[{detInv, detInvFn,m1,m2,dat, Zval, Vratio,areaFactorCorrect},
  (* convert prior to mchirp eta *)
  detInv = Det[{{D[mc[m1,m2],m1], D[mc[m1,m2],m2]}, {D[eta[m1,m2],m1], D[eta[m1,m2],m2]}}]//Simplify;
  detInvFn = Function @@{{m1,m2}, detInv};

  (* Do integral by explicit sampling, assuming the *sample* points are random in mc, eta, mod some boundary [WE MAY CHANGE THIS LATER] *)
  (* AND we must factor for the CUTOFF PROVIDED BY THE SMALL INTEGRATION REGION *)
  Print[" WARNING: Discrete evidence calculation does not account for small volume of m1,m2 covered by samples. Need to calculate ellipsoid volume MANUALLY, e.g., by covariance ellipsoid "];
  areaFactorCorrect= Pi Det[ROSSigma[ m1m2ZList[[All, {1,2}]] ] ];
  Zval = areaFactorCorrect*(Plus @@ (dat ={mc[#[[1]], #[[2]]], eta[#[[1]], #[[2]]], pdfM1M2Prior[#[[1]],#[[2]]] /detInvFn[#[[1]],#[[2]] ]  Exp[#[[3]]]}&/@m1m2ZList)[[All,-1]]);
  Vratio = Zval/Exp[Max[m1m2ZList[[All,3]]]];
  {Log[Zval], Log[Vratio], ListPointPlot3D[dat, PlotRange->All]}
]


ROSSigma[dat_] := Module[{datAv},
  datAv = Mean[dat];
  
  (Plus @@ (Outer[Times, # - datAv, # - datAv] & /@ 
       Take[dat, All]))/(Length[dat] - 1)
  ]
