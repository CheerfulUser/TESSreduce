self.get_ref()
self.make_mask(maglim=18,strapsize=4,scale=mask_scale)#Source_mask(ref,grid=0)
self.background()
self.flux = flux - self.bkg
self.centroids_DAO()
#reset for difference imaging 
self.flux = tpf.flux.value
self.shift_images()
self.flux -= self.ref
self.make_mask(maglim=18,strapsize=4,scale=mask_scale*.5)#Source_mask(ref,grid=0)
# make sure the target is well masked
m_tar = np.zeros_like(self.mask,dtype=int)
m_tar[self.size//2,self.size//2]= 1
m_tar = convolve(m_tar,np.ones((5,5)))
self.mask = self.mask | m_tar
self.background()
self.flux -= self.bkg
self.lc, self.sky = self.diff_lc(plot=True,tar_ap=tar_ap,sky_in=sky_in,sky_out=sky_out)