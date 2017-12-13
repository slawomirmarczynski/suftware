# install.packages('asaur')  # This is where all the data sets are
library(asaur)
library(survival)

write.table(ChanningHouse, file="~/github/15_deft/data/asaur/ChanningHouse.txt", quote=FALSE)

#head(ChanningHouse)

#ChanningHouse <- within(ChanningHouse, {
#  + entryYears <- entry/12
#  + exitYears <- exit/12})
#ChanningMales <- ChanningHouse[ChanningHouse$sex == "male"]

#result.km <- survfit(Surv(entryYears, exitYears, cens, type="counting") ~ 1, data=ChanningMales)
#plot(result.km, xlim=c(64,101), xlab="Age", ylab="Survival probability", conf.int=F)

