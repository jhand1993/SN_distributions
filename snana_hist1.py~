import numpy as np
import matplotlib.pyplot as plt
import array
import math
from astropy import cosmology as cosmo
hdir='/project/rkessler/jaredhand/SIMS'
named=['/project/rkessler/SN/SNDATA_ROOT/INTERNAL/PS1/S16Analysis/DATA/SALT2mu/SALT2mu_fitoptg0.fitres','/project/rkessler/SN/SNDATA_ROOT/INTERNAL/PS1/S16Analysis/DATA/SALT2mu/SALT2mu_fitoptg0.fitres']
surv=[15,99,1,4]
survname=['PS1','Low-z','SDSS','SNLS']

names1=['/project/rkessler/jaredhand/SIMS/SIMFIT_PS1_1KMS0/JSH_1KMS0_G10_PS1/FITOPT000.FITRES','/project/rkessler/jaredhand/SIMS/SIMFIT_LOWZ_1KMS0/JSH_1KMS0_G10_LOWZ/FITOPT000.FITRES','/project/rkessler/jaredhand/SIMS/SIMFIT_SDSS_1KMS0/JSH_1KMS0_G10_SDSS/FITOPT000.FITRES','/project/rkessler/jaredhand/SIMS/SIMFIT_SNLS_1KMS0/JSH_1KMS0_G10_SNLS/FITOPT000.FITRES']
#names2=['../SIMS/PS1_SINGLEu2/DS_smearC11_PS1_GRIDu_single/FITOPT000.FITRES','../SIMS/LOWZ_GRIDu_single/DS_ABCD_smearC11_LOW_GRID_single2/FITOPT000.FITRES','../SIMS/SDSS_GRIDu_single/DS_ABCD_smearC11_SDSS_GRID_single/FITOPT000.FITRES','../SIMS/SNLS_GRIDu_single/DS_ABCD_smearC11_SNLS_GRID_single/FITOPT000.FITRES']
names2=['/project/rkessler/jaredhand/SIMS/SIMFIT_PS1_1KMS0/JSH_1KMS0_G10_PS1/FITOPT000.FITRES','/project/rkessler/jaredhand/SIMS/SIMFIT_LOWZ_1KMS0/JSH_1KMS0_G10_LOWZ/FITOPT000.FITRES','/project/rkessler/jaredhand/SIMS/SIMFIT_SDSS_1KMS0/JSH_1KMS0_G10_SDSS/FITOPT000.FITRES','/project/rkessler/jaredhand/SIMS/SIMFIT_SNLS_1KMS0/JSH_1KMS0_G10_SNLS/FITOPT000.FITRES']
files=['ps1','lowz','sdss','snls']
header1d=[1,1,1,1]
header1s=[5,5,5,5]
zm1=[0,0,0.05,0.1]
zm2=[0.8,0.1,0.5,1.0]
for ii in range(0,4):
    print named[0]
    data1=np.genfromtxt(named[0],skip_header=header1d[ii],names=True,comments='#')
    cid=np.genfromtxt(named[0],skip_header=header1d[ii],usecols=(1),comments='#',dtype='str')[1:]
    z1 = data1['zCMB'].astype(float)
    SNRMAX11=data1['SNRMAX1'].astype(float)
    x11 = data1['x1'].astype(float)
    c1 = data1['c'].astype(float)
    NDOF1=data1['NDOF'].astype(float)
    TGAPMAX1=data1['TGAPMAX'].astype(float)
    FITPROB1=data1['FITPROB'].astype(float)
    PKMJD1=data1['PKMJD'].astype(float)
    RA1=data1['RA'].astype(float)
    DEC1=data1['DECL'].astype(float)
    idsurvey=data1['IDSURVEY'].astype(float)
    NDOF1=data1['NDOF'].astype(float)
    MB1E=data1['cERR'].astype(float)
    PKMJDE1=data1['PKMJDERR'].astype(float)

    data1=np.genfromtxt(names1[ii],skip_header=header1s[ii],names=True,comments='#')
    cid2=np.genfromtxt(names1[ii],skip_header=header1s[ii],usecols=(1),comments='#',dtype='str')[1:]
    z2 = data1['zCMB'].astype(float)
    SNRMAX12=data1['SNRMAX1'].astype(float)
    x12 = data1['x1'].astype(float)
    c2 = data1['c'].astype(float)
    NDOF2=data1['NDOF'].astype(float)
    #TGAPMAX2=data1['TGAPMAX'].astype(float)
    FITPROB2=data1['FITPROB'].astype(float)
    PKMJD2=data1['PKMJD'].astype(float)
    #RA2=data1['RA'].astype(float)
    #DEC2=data1['DECL'].astype(float)
    idsurvey2=data1['IDSURVEY'].astype(float)
    NDOF2=data1['NDOF'].astype(float)
    MB2E=data1['cERR'].astype(float)
    PKMJDE2=data1['PKMJDERR'].astype(float)
    
    data1=np.genfromtxt(names2[ii],skip_header=header1s[ii],names=True,comments='#')
    cid3=np.genfromtxt(names2[ii],skip_header=header1s[ii],usecols=(1),comments='#',dtype='str')[1:]
    z3 = data1['zCMB'].astype(float)
    SNRMAX13=data1['SNRMAX1'].astype(float)
    x13 = data1['x1'].astype(float)
    c3 = data1['c'].astype(float)
    NDOF3=data1['NDOF'].astype(float)
    #TGAPMAX3=data1['TGAPMAX'].astype(float)
    FITPROB3=data1['FITPROB'].astype(float)
    PKMJD3=data1['PKMJD'].astype(float)
    #RA3=data1['RA'].astype(float)
    #DEC3=data1['DECL'].astype(float)
    idsurvey3=data1['IDSURVEY'].astype(float)
    NDOF3=data1['NDOF'].astype(float)
    MB3E=data1['cERR'].astype(float)
    PKMJDE3=data1['PKMJDERR'].astype(float)
    
    

    if (surv[ii]!=99): xx=((np.absolute(c1)<0.3)&(np.absolute(x11)<3)&(FITPROB1>0.001)&(idsurvey==surv[ii]))
    if (surv[ii]==99): xx=((np.absolute(c1)<0.3)&(np.absolute(x11)<3)&(FITPROB1>0.001)&(idsurvey!=15)&(idsurvey!=1)&(idsurvey!=4)&(z1<0.1))

    z1=z1[xx]
    SNRMAX11=SNRMAX11[xx]
    x11=x11[xx]
    c1=c1[xx]
    NDOF1=NDOF1[xx]
    TGAPMAX1=TGAPMAX1[xx]
    RA1=RA1[xx]
    PKMJDE1=PKMJDE1[xx]
    MB1E=MB1E[xx]

    PKMJD2 = PKMJD2.astype(float)
    PKMJDE2 = PKMJDE2.astype(float)
    
    z2 = z2.astype(float)
    SNRMAX12=SNRMAX12.astype(float)
    x12 = x12.astype(float)
    c2 = c2.astype(float)
    NDOF2=NDOF2.astype(float)
    FITPROB2=FITPROB2.astype(float)
    MB2E=MB2E.astype(float)
    
    xx=((np.absolute(c2)<0.3)&(np.absolute(x12)<3)&(FITPROB2>0.001))
    z2=z2[xx]
    SNRMAX12=SNRMAX12[xx]
    x12=x12[xx]
    c2=c2[xx]
    NDOF2=NDOF2[xx]
    FITPROB2=FITPROB2[xx]
    PKMJD2=PKMJD2[xx]
    PKMJDE2=PKMJDE2[xx]
    MB2E=MB2E[xx]
    
    
    z3 = z3.astype(float)
    SNRMAX13=SNRMAX13.astype(float)
    PKMJDE3 = PKMJDE3.astype(float)
    x13 = x13.astype(float)
    c3 = c3.astype(float)
    NDOF3=NDOF3.astype(float)
    FITPROB3=FITPROB3.astype(float)
    PKMJD3=PKMJD3.astype(float)
    
    MB3E=MB3E.astype(float)

    xx=((np.absolute(c3)<0.3)&(np.absolute(x13)<3)&(FITPROB3>0.001))
    z3=z3[xx]
    SNRMAX13=SNRMAX13[xx]
    x13=x13[xx]
    c3=c3[xx]
    NDOF3=NDOF3[xx]
    PKMJD3=PKMJD3[xx]
    PKMJDE3=PKMJDE3[xx]
    MB3E=MB3E[xx]
    
 


    col=['b','g','r','c','m','y','r','g','m','b','g']

    pos=[]



    for i in range(1,299):
        pos.append(i/10.0-5)
    pos2=[]

    for i in range(1,299):
        pos2.append(0.0/100.0+.11)

    plt.figure(1)



    fig, ax = plt.subplots(4,2)


    n, bins, patches = ax[0,0].hist(z1, bins=10,range=[zm1[ii],zm2[ii]], color='white', alpha=0.25,linewidth=0)
    ax[0,0].set_xlabel('z')
    ax[0,0].set_ylabel('#')
    ax[0,0].errorbar((bins[:-1]+bins[1:])/2.0, n, yerr=np.sqrt(n), fmt='ko', ecolor='k') 
    weights=z2*0.0+float(len(z1))/float(len(z2))
    weights3=z3*0.0+float(len(z1))/float(len(z3))

    print 'weights', weights
    n2, bins2, patches2 = ax[0,0].hist(z2, weights=weights,bins=10,range=[zm1[ii],zm2[ii]], histtype='step', alpha=0.85, color='r',linewidth=2)
    n3, bins3, patches3 = ax[0,0].hist(z3, weights=weights3,bins=10,range=[zm1[ii],zm2[ii]], histtype='step', alpha=0.85, color='g',linewidth=2)
    print np.sum(n), np.sum(n2), np.sum(n3)

    n, bins, patches = ax[1,0].hist(PKMJDE1, bins=15,range=[0,3], color='white', alpha=0.25,linewidth=0)
    ax[1,0].errorbar((bins[:-1]+bins[1:])/2.0, n, yerr=np.sqrt(n), fmt='ko', ecolor='k')
    ax[1,0].set_xlabel(r"$\sigma_{pkMJD}$")
    ax[1,0].set_ylabel('#')
    n2, bins2, patches2 = ax[1,0].hist(PKMJDE2,weights=weights, bins=15,range=[0,3], histtype='step', alpha=0.85,color='r',linewidth=2)
    n2, bins2, patches2 = ax[1,0].hist(PKMJDE3,weights=weights3, bins=15,range=[0,3], histtype='step', alpha=0.85,color='g',linewidth=2)
    



    n, bins, patches = ax[0,1].hist(SNRMAX11, bins=15,range=[0,70], color='white', alpha=0.25,linewidth=0)
    ax[0,1].errorbar((bins[:-1]+bins[1:])/2.0, n, yerr=np.sqrt(n), fmt='ko', ecolor='k')

    ax[0,1].set_ylabel('#')
    ax[0,1].set_xlabel('SNR')
    n2, bins2, patches2 = ax[0,1].hist(SNRMAX12, weights=weights, color='r',bins=15,range=[0,70], histtype='step', alpha=0.85,linewidth=2)
    n3, bins2, patches2 = ax[0,1].hist(SNRMAX13, weights=weights3, color='green',bins=15,range=[0,70], histtype='step', alpha=0.85,linewidth=2)

    n, bins, patches = ax[1,1].hist(MB1E, bins=10,range=[0,0.1], color='white', alpha=0.25,linewidth=0)
    ax[1,1].errorbar((bins[:-1]+bins[1:])/2.0, n, yerr=np.sqrt(n), fmt='ko', ecolor='k')

    ax[1,1].set_xlabel(r"$\sigma_c$")
    ax[1,1].set_ylabel('#')
    n2, bins2, patches2 = ax[1,1].hist(MB2E,weights=weights,color='r', bins=10,range=[0,.10], histtype='step', alpha=0.85,linewidth=2)
    n2, bins2, patches2 = ax[1,1].hist(MB3E,weights=weights3,color='g', bins=10,range=[0,.10], histtype='step', alpha=0.85,linewidth=2)
    ax[1,1].set_ylim(0,130)



    n, bins, patches = ax[2,0].hist(c1, bins=15,range=[-.4,.4], facecolor='white', alpha=0.25,linewidth=0)
    ax[2,0].errorbar((bins[:-1]+bins[1:])/2.0, n, yerr=np.sqrt(n), fmt='ko', ecolor='k')
    n2, bins2, patches2 = ax[2,0].hist(c2,weights=weights,color='r', bins=15,range=[-.4,.4], histtype='step', alpha=0.85,linewidth=2)
    
    n3, bins3, patches3 = ax[2,0].hist(c3,weights=weights3,color='g', bins=15,range=[-.4,.4], histtype='step', alpha=0.85,linewidth=2)
    
    print np.sum(n), np.sum(n2), np.sum(n3)

    ax[2,0].set_xlabel('c')
    ax[2,0].set_ylabel('#')
    ax[2,0].set_xlim(-.3,.3)
    
    n, bins, patches = ax[2,1].hist(x11, bins=15,range=[-4,4], color='white', alpha=0.25,linewidth=0)
    ax[2,1].errorbar((bins[:-1]+bins[1:])/2.0, n, yerr=np.sqrt(n), fmt='ko', ecolor='k')
    
    #ax[2,1].xlabel='z'
    #ax[2,1].ylabel='#'
    n2, bins2, patches2 = ax[2,1].hist(x12, weights=weights,color='r',bins=15,range=[-4,4], histtype='step', alpha=0.85,linewidth=2)
    n3, bins3, patches3 = ax[2,1].hist(x13, weights=weights3,color='green',bins=15,range=[-4,4], histtype='step', alpha=0.85,linewidth=2)
    
    #ax[3,0].errorbar(z1, c1, yerr=c1*0+.005, fmt='ko', ecolor='k')
    #ax[3,0].xlim([0,0.7])
    #ax[3,0].ylim([-.3,.3])
    #ax[3,0].ylabel('c')
    #ax[3,0].xlabel('z ')
    #ax[3,0].errorbar(z1, c1, yerr=c1*0+.005, fmt='bo', ecolor='b')
    ax[2,1].set_xlabel('x1')
    ax[2,1].set_ylabel('#')
    ax[2,1].set_xlim(-3,3)
    
    bins = np.linspace(zm1[ii], zm2[ii], 11)
    #l1=plt.legend([p1],['D15 Data'],loc='lower left',prop={'size':16})
    digitized = np.digitize(z2, bins)
    bin_means = [np.median(c2[digitized == i]) for i in range(0, len(bins))]
    bin_z = [np.median(z2[digitized == i]) for i in range(0, len(bins))]
    bin_std = [np.std(c2[digitized == i])/np.sqrt(len(c2[digitized == i])) for i in range(0, len(bins))]
    #ax[3,0].errorbar(bin_z, bin_means, yerr=bin_std, fmt='ko', ecolor='g', color='g',label='D15 Sim')
    line, = ax[3,0].plot(bin_z,bin_means, lw=4,color='red',alpha=.85)
    
    digitized = np.digitize(z3, bins)
    bin_means = [np.median(c3[digitized == i]) for i in range(0, len(bins))]
    bin_z = [np.median(z3[digitized == i]) for i in range(0, len(bins))]
    bin_std = [np.std(c3[digitized == i])/np.sqrt(len(c3[digitized == i])) for i in range(0, len(bins))]
    #ax[3,0].errorbar(bin_z, bin_means, yerr=bin_std, fmt='ko', ecolor='g', color='g',label='D15 Sim')                                                                                                      
    line, = ax[3,0].plot(bin_z,bin_means, lw=4,color='green',alpha=.85, label='Sim')

    bins = np.linspace(zm1[ii], zm2[ii], 11)
    
    digitized = np.digitize(z1, bins)
    bin_means = [np.median(c1[digitized == i]) for i in range(0, len(bins))]
    bin_z = [np.median(z1[digitized == i]) for i in range(0, len(bins))]
    bin_std = [np.std(c1[digitized == i])/np.sqrt(len(c1[digitized == i])) for i in range(0, len(bins))]
    ax[3,0].errorbar(bin_z, bin_means, yerr=bin_std, fmt='ko', ecolor='black', color='black',label='Data')
    
    
    ax[3,0].legend(loc='lower left',prop={'size':8})
    ax[3,0].set_ylabel('c')
    ax[3,0].set_xlabel('z')
    ax[3,0].set_xlim(zm1[ii],zm2[ii])
    
    
    #l1=plt.legend([p1],['D15 Data'],loc='lower left',prop={'size':16})                                                                                     
    digitized = np.digitize(z2, bins)
    bin_means = [np.median(x12[digitized == i]) for i in range(0, len(bins))]
    bin_z = [np.median(z2[digitized == i]) for i in range(0, len(bins))]
    bin_std = [np.std(x12[digitized == i])/np.sqrt(len(x12[digitized == i])) for i in range(0, len(bins))]
    #ax[3,1].errorbar(bin_z, bin_means, yerr=bin_std, fmt='ko', ecolor='g', color='g',label='D15 Sim')
    line, = ax[3,1].plot(bin_z,bin_means, lw=4,color='red',alpha=.85)
    
    digitized = np.digitize(z3, bins)
    bin_means = [np.median(x13[digitized == i]) for i in range(0, len(bins))]
    bin_z = [np.median(z3[digitized == i]) for i in range(0, len(bins))]
    bin_std = [np.std(x13[digitized == i])/np.sqrt(len(x13[digitized == i])) for i in range(0, len(bins))]
    #ax[3,1].errorbar(bin_z, bin_means, yerr=bin_std, fmt='ko', ecolor='g', color='g',label='D15 Sim')                                                                                         
    
    line, = ax[3,1].plot(bin_z,bin_means, lw=4,color='green',alpha=.85)
    
    digitized = np.digitize(z1, bins)
    bin_means = [np.median(x11[digitized == i]) for i in range(0, len(bins))]
    bin_z = [np.median(z1[digitized == i]) for i in range(0, len(bins))]
    bin_std = [np.std(x11[digitized == i])/np.sqrt(len(x11[digitized == i])) for i in range(0, len(bins))]
    ax[3,1].errorbar(bin_z, bin_means, yerr=bin_std, fmt='ko', ecolor='black', color='black',label='D15 Data')


    #ax[3,1].legend(loc='upper right',prop={'size':5})
    ax[3,1].set_ylabel('x1')
    ax[3,1].set_xlabel('z')
    ax[3,1].set_xlim(zm1[ii],zm2[ii])
    
    ax[0,0].text(.7,50,"(a)",fontdict={'fontsize':10})
    ax[0,1].text(70,60,"(b)",fontdict={'fontsize':10})
    ax[1,0].text(3.5,90,"(c)",fontdict={'fontsize':10})
    ax[1,1].text(.12,100,"(d)",fontdict={'fontsize':10})
    ax[2,0].text(.3,90,"(e)",fontdict={'fontsize':10})
    ax[2,1].text(3,70,"(f)",fontdict={'fontsize':10})
    ax[3,0].text(.7,.05,"(g)",fontdict={'fontsize':10})
    ax[3,1].text(.7,1.5,"(h)",fontdict={'fontsize':10})

    plt.tight_layout()
    plt.show()


    plt.savefig('snana_hist_'+files[ii]+'.png')
    plt.cla()
# stop
plt.figure(1)
bins = np.linspace(18,24,12)
digitized = np.digitize(mb3, bins)
bin_means = [np.median(SNRMAX13[digitized == i]) for i in range(0, len(bins))]
bin_z = [np.median(mb3[digitized == i]) for i in range(0, len(bins))]
bin_std = [np.std(SNRMAX13[digitized == i])/np.sqrt(len(SNRMAX13[digitized == i])) for i in range(0, len(bins))]
#ax[3,1].errorbar(bin_z, bin_means, yerr=bin_std, fmt='ko', ecolor='g', color='g',label='D15 Sim')                                                                               

line, = plt.plot(bin_z,bin_means, lw=4,color='green',alpha=.85)

print mb1
print SNRMAX11

digitized = np.digitize(mb1, bins)
bin_means = [np.median(SNRMAX11[digitized == i]) for i in range(0, len(bins))]
bin_z = [np.median(mb1[digitized == i]) for i in range(0, len(bins))]
bin_std = [np.std(SNRMAX11[digitized == i])/np.sqrt(len(SNRMAX11[digitized == i])) for i in range(0, len(bins))]
plt.errorbar(bin_z, bin_means, yerr=bin_std, fmt='ko', ecolor='b', color='b',label='D15 Data',alpha=.5)
print bin_z
print bin_means

plt.xlim(16,24)
plt.ylim(0,200)
plt.xlabel('mb')
plt.ylabel('SNRMAX1')
plt.tight_layout()
plt.show()


plt.savefig('snana_hist_check.png')

# asdf
