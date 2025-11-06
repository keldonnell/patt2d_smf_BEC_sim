import analytic_predictors as ap

p_th = (2 / (150)) * 1.14e-5
p0 = 1.53e-7
gambar = 1.14e-5
seed = 1e-6
print(ap.analytic_delay_time(p0, p_th, gambar, seed))
#print((2 / (150)) * 1.14e-5)