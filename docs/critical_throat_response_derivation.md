# Single throat response in the linear PNM

This note derives the fixed-network response readout used by TopoFlow. It applies
the Sherman–Morrison matrix identity to the declared linear hydraulic reference;
it introduces no new physical law or experimental result.

Let the network Laplacian be \(L=BGB^T\), where the columns of \(B\) are oriented
throat incidence vectors and \(G\) contains their conductances. Divide pore
pressures into free values \(p_F\) and prescribed Dirichlet values \(p_D\).
On the retained conducting components,

\[
A p_F=-L_{FD}p_D,\qquad A=L_{FF}.
\]

The derivation assumes that \(A\) is invertible after the declared component and
boundary handling. Let a selected throat have conductance \(g>0\), full incidence
vector \(b\), free-node incidence \(a=b_F\), and baseline pressure drop
\(v=b^Tp\). The full-network dissipated power is

\[
P=p^TLp=\sum_j g_j(b_j^Tp)^2>0.
\]

Define its baseline dissipation fraction and Dirichlet edge leverage by

\[
s=\frac{g v^2}{P},\qquad \ell=g a^TA^{-1}a.
\]

Changing only this throat's conductance to \(rg\) gives
\(\delta g=(r-1)g\) and \(A'=A+\delta g\,aa^T\). The boundary forcing changes
consistently if an endpoint has prescribed pressure. Applying the
[Sherman–Morrison identity](https://doi.org/10.1214/aoms/1177729893) gives

\[
p'_F-p_F=-\frac{\delta g\,A^{-1}a\,v}{1+\delta g\,a^TA^{-1}a},
\qquad
v'=\frac{v}{1+\delta g\,a^TA^{-1}a}.
\]

The baseline free pressures are stationary points of the dissipation at fixed
boundary values. Substituting the pressure change into the quadratic energy
therefore yields

\[
P'-P=\frac{\delta g\,v^2}{1+\delta g\,a^TA^{-1}a}.
\]

With the same two-terminal pressure difference, viscosity and image dimensions,
Darcy permeability is proportional to power. Thus

\[
1-\frac{k'}{k}
=-\frac{(r-1)s}{1+(r-1)\ell}
=\frac{(1-r)s}{1-(1-r)\ell}.
\]

In the cylindrical-throat closure, narrowing diameter by fraction \(q\) while
holding length fixed gives \(r=(1-q)^4\). The tested doses keep \(r>0\).
Complete deletion or a singular updated network requires separate connectivity
handling; the inverse-update expression must not be applied through a zero
denominator.

The identity is exact for the stated linear network when \(s\) and \(\ell\)
come from its solved flow and matrix inverse. The surrogate uses learned
estimates instead. Its softmax importance and bounded leverage head enforce
\(\hat s\geq0\) and \(0\leq\hat\ell\leq1-\hat s\). These bounds make the
predicted loss bounded and monotone for \(0\leq r\leq1\), where the denominator
is nonzero; they do not establish calibration or physical validity.

The committed [discovery verification](../experiments/topoflow_discovery/verification.json)
records 966 fresh intervention solves and a maximum fractional discrepancy of
approximately \(4.5\times10^{-8}\) from the exact update. This checks the
implementation under its closure. It is not independent validation against
voxel-resolved flow or measured permeability.

The paper's separately trained global importance head is distinct from the
neighborhood-normalized attention coefficients inside a GAT layer.

Reference: J. Sherman and W. J. Morrison, *The Annals of Mathematical Statistics*,
21(1), 124–127, 1950. DOI: 10.1214/aoms/1177729893.
