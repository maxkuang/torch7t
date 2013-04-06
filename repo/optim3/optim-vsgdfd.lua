-- d-vsgd-fd
-- author: Sixin Zhang and Tom Schaul

require 'torch'
require 'optim'

local function outdet(v,mv,vv)
   -- diff =  |v-mv| - 2sqrt(vv-mv^2)
   -- return diff > 0
   local tmp = v:clone():fill(0)
   local tmp2 = v:clone():fill(0)
   tmp:copy(v)
   tmp:add(-1,mv)
   tmp:abs()
   
   tmp2:copy(mv)
   tmp2:cmul(mv)
   tmp2:mul(-1)
   tmp2:add(vv)
   tmp2:mul(-1)
   tmp2:abs()
   
   tmp2:sqrt()
   tmp:add(-2,tmp2) -- diff

   tmp:shrink(0,0)
   tmp:mul(-1) -- -diff
   tmp:shrink(0,1)
   -- shrink(t,v) = (x<t?v:x)

   return tmp
end

function optim.vsgdfd(opfunc, x, state)

   local state = state or {}
   state.eps = state.eps or 1e-6
   state.eps2 = state.eps2 or 1e-12
   state.nevals = state.nevals or 0
   state.initSamples = state.initSamples or math.max(1,torch.sqrt(x:nElement()))
   state.slowStartConst = state.slowStartConst or math.max(2,x:nElement()/10)

   state.clr = state.clr or x:clone():fill(0)

   state.gbars = state.gbars or x:clone():fill(0)
   state.vbars = state.vbars or x:clone():fill(0)
   state.hbars = state.hbars or x:clone():fill(0)
   state.h2bars = state.h2bars or x:clone():fill(1)
   state.taus = state.taus or x:clone():fill(1)

   state.dfdx = state.dfdx or x:clone():fill(0)
   state.dfdx2 = state.dfdx2 or x:clone():fill(0)
   state.diagh = state.diagh or x:clone():fill(0)

   state.invtaus = state.invtaus or x:clone():fill(1)
   state.decay = state.decay or x:clone():fill(0)

   local fx,dfdx = opfunc(x)
   state.dfdx:copy(dfdx)
   state.dfdx:add(state.eps)

   state.dfdx2:copy(state.dfdx)
   state.dfdx2:cmul(state.dfdx)
   state.dfdx2:add(state.eps2)

   state.diagh:copy(x)
   if state.nevals == 0 then
      state.gbars:copy(state.dfdx)
   end
   state.diagh:add(state.gbars) -- temp diagh for x+gbar
   local fx1,dfdx1 = opfunc(state.diagh) -- opfunc should bak original x
   dfdx1:add(state.eps)
   state.diagh:copy(dfdx1)
   state.diagh:add(-1,state.dfdx)
   state.diagh:cdiv(state.gbars)
   state.diagh:abs()  -- diagh = abs(dfdx - dfdx_2)/abs(eps+state.gbars)
   state.diagh:add(state.eps)

   if state.initSamples > state.nevals then
      -- init
      state.gbars:add(1/state.initSamples,state.dfdx)
      state.vbars:add(1/state.initSamples,state.dfdx2)
      state.hbars:add(1/state.initSamples,state.diagh)
      state.diagh:cmul(state.diagh) -- squaring
      state.h2bars:add(1/state.initSamples,state.diagh)
   else
      -- init
      if state.initSamples + 1 > state.nevals then
	 state.vbars:mul(state.slowStartConst)
	 state.hbars:mul(state.slowStartConst)
	 state.h2bars:mul(state.slowStartConst*state.slowStartConst)
	 state.decay:copy(state.gbars)
	 state.decay:cmul(state.gbars)
	 state.decay:cdiv(state.vbars)
      end

      -- taus
      state.decay:mul(-1)
      state.decay:add(1)

      state.taus:cmul(state.decay)
      state.taus:add(1)
      
      -- outlier detect
      state.taus:add(outdet(state.dfdx,state.gbars,state.vbars))
      state.taus:add(outdet(state.diagh,state.hbars,state.h2bars))

      -- decay
      state.invtaus:fill(1)
      state.invtaus:cdiv(state.taus)
      state.decay:copy(state.invtaus)
      state.decay:mul(-1)
      state.decay:add(1)

      -- averaging
      state.gbars:cmul(state.decay)
      state.gbars:addcmul(1, state.invtaus, state.dfdx)
      
      state.vbars:cmul(state.decay)
      state.vbars:addcmul(1, state.invtaus, state.dfdx2)

      state.hbars:cmul(state.decay)
      state.hbars:addcmul(1, state.invtaus, state.diagh)

      state.diagh:cmul(state.diagh) -- squaring
      state.h2bars:cmul(state.decay)
      state.h2bars:addcmul(1, state.invtaus, state.diagh)

      -- update
      state.decay:copy(state.gbars)
      state.decay:cmul(state.gbars)
      state.decay:cdiv(state.vbars)

      state.clr:copy(state.decay)
      state.clr:cmul(state.hbars)
      state.clr:cdiv(state.h2bars)
      x:addcmul(-1, state.clr, state.dfdx)
   end

   state.nevals = state.nevals + 1

   -- return x after, f(x) before, optimization
   return x,{fx}
end

