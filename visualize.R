library(grid)
library(data.table)
library(gridExtra)
library(grid)
library(tidyr)
library(dplyr)
library(gt)
library(gtExtras)
library(ggplot2)
library(ggalt)
library(scales)
library(ggpubr)

mse_data <- fread("results/test_mse.csv")
mse_data <- data.frame(mse_data)

ggplot(mse_data, aes(x=it,y=mse, color="#0091ff")) +
  scale_y_continuous(trans='log10',
                     breaks=trans_breaks('log10', function(x) 10^x),
                     labels=trans_format('log10', math_format(10^.x))) +
  geom_point(show.legend = FALSE)

aprox_data <- fread("results/test_predict.csv")
aprox_data <- data.frame(aprox_data)

ggplot(aprox_data, aes(x=x)) +
  geom_line(aes(y=y_predict, colour="przewidywana")) +
  geom_line(aes(y=y, colour="dokładna")) +
  scale_color_manual(name='Legenda', values=c('przewidywana'='red', 'dokładna'='blue'))

