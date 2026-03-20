"""1. KDPI is used to quantify the quality of deceased donor kidneys relative to other recovered kidneys. SRTR has a report detailing its usage.
KDPI Reference: https://optn.transplant.hrsa.gov/media/j34dm4mv/kdpi_guide.pdf

2. eGFR is commonly used for renal functions. Currently, the 2021 CKD-EPI equation is used. The equation is derived from the following paper.
eGFR Reference: https://www.nejm.org/doi/full/10.1056/NEJMoa2102953"""


#KDRI Calculate
#Require External Mapping File
KDPI_sca = read.csv('data/KDRI_Scale_Fac.csv')
KDPI_map = read.csv('data/KDPI_Mapping_090424_updateto2023.csv')
tx_kdpi <- tx_ki %>% mutate(KDRI_x = 0.0128*(DON_AGE-40)-0.0194*(DON_AGE-18)*(DON_AGE<18)+0.0107*(DON_AGE-50)*(DON_AGE>50)-0.0464*(DON_HGT_CM-170)/10-
                                     0.0199*(DON_WGT_KG-80)/5+0.179*(DON_RACE==16)+0.126*(DON_HTN==1)+0.13*(DON_HIST_DIAB%in%c(2,3,4,5))+0.0881*(DON_CAD_DON_COD==2)+
                                     0.22*(DON_CREAT-1)-0.209*(DON_CREAT-1.5)*(DON_HIGH_CREAT==1)+0.24*(DON_ANTI_HCV=='P')+0.133*(DON_NON_HR_BEAT=='Y'),
                            KDRI_unsca = exp(KDRI_x),
                            Year = as.numeric(substring(CAN_LISTING_DT,1,4)))
#Scaling KDRI
for (i in 1:nrow(tx_kdpi)){
   tx_kdpi[i,'KDRI'] = tx_kdpi[i,'KDRI_unsca']/KDPI_sca[which(KDPI_sca$Year==unlist(tx_kdpi[i,'Year'])),'Scale_Fac']
   tx_kdpi[i,'KDPI'] = KDPI_map[which((unlist(tx_kdpi[i,'KDRI'])<=KDPI_map$up)&(unlist(tx_kdpi[i,'KDRI'])>KDPI_map$low)&(KDPI_map$Year==unlist(tx_kdpi[i,'Year']))),'KDPI']
}

#Note that eGFR can be calculated whenever a creatinine measurement is available.
#The following code shows the eGFR calculation for eGFR immediately before transplant.
tx_kdpi$REC_eGFR = ifelse(tx_kdpi$CAN_GENDER == 'M',142*pmin(tx_kdpi$REC_CREAT/0.9,1)^(-0.302)*pmax(tx_kdpi$REC_CREAT/0.9,1)^(-1.2)*
0.9938^(tx_kdpi$REC_AGE_AT_TX),142*pmin(tx_kdpi$REC_CREAT/0.7,1)^(-0.241)*pmax(tx_kdpi$REC_CREAT/0.7,1)^(-1.2)*0.9938^(tx_kdpi$REC_AGE_AT_TX)*1.012)


#Here we calculate the rate for each calendar month. We use the year 2010 as the starting point for illustration. The unit is per person-month.
dt <- cand_kipa %>%
      filter(WL_ORG=='KI',as.numeric(substr(CAN_REM_DT,1,4))>=2010 | is.na(CAN_REM_DT), !is.na(CAN_LISTING_DT)) %>%
      mutate(start_month = pmax(as.numeric(substring(CAN_LISTING_DT,6,7))+12*(as.numeric(substring(CAN_LISTING_DT,1,4))-2010),0),
             end_month = as.numeric(substring(CAN_REM_DT,6,7))+12*(as.numeric(substr(CAN_REM_DT,1,4))-2010),
             start_week = pmax(as.numeric(difftime(CAN_LISTING_DT,as.Date('2010-01-01'),units = 'week')),0),
             end_week = as.numeric(difftime(CAN_REM_DT,as.Date('2010-01-01'),units = 'week')),
             day_start_month = as.numeric(substring(CAN_LISTING_DT,9,10)),
             day_end_month = as.numeric(substring(CAN_REM_DT,9,10)))
Txp_rate_plot_dt <- data.frame(month = c(1:max(dt$end_month,na.rm = T)),Txp_rate = c(NA))
for (i in 1:(nrow(Txp_rate_plot_dt)-1)) {
    days_at_month <- ifelse(i%%12 %in% c(1,3,5,7,8,10,0), 31,
    ifelse(i%%12 == 2 & (i%/%12)%%4==0, 29,
    ifelse(i%%12 == 2, 28, 30)))
    dt$age_cur = dt$CAN_AGE_AT_LISTING + floor(i/12) + 2010 - as.numeric(substr(dt$CAN_LISTING_DT,1,4))
    temp <- dt %>%
            filter(start_month<=i,end_month>=i|is.na(end_month)) %>%
            mutate(person_month = ifelse(is.na(end_month),1,
                                         ifelse(start_month==end_month,(day_end_month-day_start_month)/days_at_month,
                                                ifelse(start_month==i,(days_at_month+1-day_start_month)/days_at_month,
                                                       ifelse(end_month==i,day_end_month/days_at_month,1)))))
                                                                                                                      
                                                                                                                      
    Txp_rate_plot_dt[i,'Txp_count'] <- nrow(dt %>% filter(CAN_REM_CD %in% c(4,15,18,19), end_month == i))
    Txp_rate_plot_dt[i,'Deceased_Txp_count'] <- nrow(dt %>% filter(CAN_REM_CD %in% c(4), end_month == i))
    Txp_rate_plot_dt[i,'Txp_free_death_count'] <- nrow(dt %>% filter(CAN_REM_CD==8, end_month == i))
    Txp_rate_plot_dt[i,'New_waitlisted_count'] <- nrow(dt %>% filter(start_month == i))
    Txp_rate_plot_dt[i,'Living_Txp_count'] <- nrow(dt %>% filter(CAN_REM_CD %in% c(15), end_month == i))
    Txp_rate_plot_dt[i,'Person_month'] <- sum(temp$'person_month')
                                                                                                         
    Txp_rate_plot_dt[i,'Txp_rate'] <- Txp_rate_plot_dt[i,'Txp_count']/Txp_rate_plot_dt[i,'Person_month']
    Txp_rate_plot_dt[i,'Deceased_Txp_rate'] <- Txp_rate_plot_dt[i,'Deceased_Txp_count']/Txp_rate_plot_dt[i,'Person_month']
    Txp_rate_plot_dt[i,'Txp_free_death_rate'] <- Txp_rate_plot_dt[i,'Txp_free_death_count']/Txp_rate_plot_dt[i,'Person_month']
    Txp_rate_plot_dt[i,'Living_Txp_rate'] <- Txp_rate_plot_dt[i,'Living_Txp_count']/Txp_rate_plot_dt[i,'Person_month']
}


#compdth means all-cause graft survival: Both graft failure and patient death count as an event.
#ptx_death means patient survival: Death is considered as an event, and graft failure is considered as censoring.
#gft means death-censored graft survival: Only graft failure is considered an event and death is considered as censoring
tx_kdpi = tx_kdpi %>% mutate(compdth = ifelse(is.na(TFL_DEATH_DT)==FALSE|is.na(TFL_GRAFT_DT)==FALSE|is.na(PERS_SSA_DEATH_DT)==F|is.na(PERS_OPTN_DEATH_DT)==F, 1, 0),
                                          ptx_death = ifelse(is.na(TFL_DEATH_DT)==FALSE|is.na(PERS_SSA_DEATH_DT)==F|is.na(PERS_OPTN_DEATH_DT)==F, 1, 0),
                                          gft = ifelse(is.na(TFL_GRAFT_DT)==FALSE, 1, 0))

#Here we calculate the rate for each calendar month. We use the year 2010 as the starting point for illustration. The unit is per person-month.
dt <- cand_kipa %>%
      filter(WL_ORG=='KI',as.numeric(substr(CAN_REM_DT,1,4))>=2010 | is.na(CAN_REM_DT), !is.na(CAN_LISTING_DT)) %>%
      mutate(start_month = pmax(as.numeric(substring(CAN_LISTING_DT,6,7))+12*(as.numeric(substring(CAN_LISTING_DT,1,4))-2010),0),
             end_month = as.numeric(substring(CAN_REM_DT,6,7))+12*(as.numeric(substr(CAN_REM_DT,1,4))-2010),
             start_week = pmax(as.numeric(difftime(CAN_LISTING_DT,as.Date('2010-01-01'),units = 'week')),0),
             end_week = as.numeric(difftime(CAN_REM_DT,as.Date('2010-01-01'),units = 'week')),
             day_start_month = as.numeric(substring(CAN_LISTING_DT,9,10)),
             day_end_month = as.numeric(substring(CAN_REM_DT,9,10)))
Txp_rate_plot_dt <- data.frame(month = c(1:max(dt$end_month,na.rm = T)),Txp_rate = c(NA))
for (i in 1:(nrow(Txp_rate_plot_dt)-1)) {
    days_at_month <- ifelse(i%%12 %in% c(1,3,5,7,8,10,0), 31,
    ifelse(i%%12 == 2 & (i%/%12)%%4==0, 29,
    ifelse(i%%12 == 2, 28, 30)))
    dt$age_cur = dt$CAN_AGE_AT_LISTING + floor(i/12) + 2010 - as.numeric(substr(dt$CAN_LISTING_DT,1,4))
    temp <- dt %>%
            filter(start_month<=i,end_month>=i|is.na(end_month)) %>%
            mutate(person_month = ifelse(is.na(end_month),1,
                                         ifelse(start_month==end_month,(day_end_month-day_start_month)/days_at_month,
                                                ifelse(start_month==i,(days_at_month+1-day_start_month)/days_at_month,
                                                       ifelse(end_month==i,day_end_month/days_at_month,1)))))
                                                                                                                      
                                                                                                                      
    Txp_rate_plot_dt[i,'Txp_count'] <- nrow(dt %>% filter(CAN_REM_CD %in% c(4,15,18,19), end_month == i))
    Txp_rate_plot_dt[i,'Deceased_Txp_count'] <- nrow(dt %>% filter(CAN_REM_CD %in% c(4), end_month == i))
    Txp_rate_plot_dt[i,'Txp_free_death_count'] <- nrow(dt %>% filter(CAN_REM_CD==8, end_month == i))
    Txp_rate_plot_dt[i,'New_waitlisted_count'] <- nrow(dt %>% filter(start_month == i))
    Txp_rate_plot_dt[i,'Living_Txp_count'] <- nrow(dt %>% filter(CAN_REM_CD %in% c(15), end_month == i))
    Txp_rate_plot_dt[i,'Person_month'] <- sum(temp$'person_month')
                                                                                                         
    Txp_rate_plot_dt[i,'Txp_rate'] <- Txp_rate_plot_dt[i,'Txp_count']/Txp_rate_plot_dt[i,'Person_month']
    Txp_rate_plot_dt[i,'Deceased_Txp_rate'] <- Txp_rate_plot_dt[i,'Deceased_Txp_count']/Txp_rate_plot_dt[i,'Person_month']
    Txp_rate_plot_dt[i,'Txp_free_death_rate'] <- Txp_rate_plot_dt[i,'Txp_free_death_count']/Txp_rate_plot_dt[i,'Person_month']
    Txp_rate_plot_dt[i,'Living_Txp_rate'] <- Txp_rate_plot_dt[i,'Living_Txp_count']/Txp_rate_plot_dt[i,'Person_month']
}

                                