-- d-vsgd
-- author: Sixin Zhang
-- thanks: Tom Schaul

require 'torch'
require 'optim'

function optim.dvsgd(opfunc, x, state)

   local eps = 1e-16

   local state = state or {}
   state.nevals = state.nevals or 0
   state.initSamples = state.initSamples or 60
   -- math.max(1,torch.sqrt(x:nElement()))
   state.slowStartConst = state.slowStartConst or math.max(1,x:nElement()/10)

   state.clr = state.clr or x:clone():fill(0)
   state.gbars = state.gbars or x:clone():fill(0)
   state.vbars = state.vbars or x:clone():fill(0)
   state.hbars = state.hbars or x:clone():fill(0)
   state.taus = state.taus or x:clone():fill(1)

   state.dfdx2 = state.dfdx2 or x:clone():fill(0)
   state.invtaus = state.invtaus or x:clone():fill(1)
   state.decay = state.decay or x:clone():fill(0)

   local fx,dfdx,diagh = opfunc(x)
   dfdx:add(eps)
   diagh:add(eps)    -- avoid 0/0
   state.dfdx2:copy(dfdx)
   state.dfdx2:cmul(dfdx)

   if state.initSamples > state.nevals then
      -- init
      state.gbars:add(1/state.initSamples,dfdx)
      state.vbars:add(1/state.initSamples,state.dfdx2)
      state.hbars:add(1/state.initSamples,diagh)
   else
      -- init
      if state.initSamples + 1 > state.nevals then
	 state.vbars:mul(state.slowStartConst)
	 state.hbars:mul(state.slowStartConst)
	 state.decay:copy(state.gbars)
	 state.decay:cmul(state.gbars)
	 state.decay:cdiv(state.vbars)
	 state.decay:shrink(eps,0)
      end

      -- taus
      state.decay:mul(-1)
      state.decay:add(1)
      state.decay:shrink(eps,0)

      state.taus:cmul(state.decay)
      state.taus:add(1)

      state.invtaus:fill(1)
      state.invtaus:cdiv(state.taus)
      state.decay:copy(state.invtaus)
      state.decay:mul(-1)
      state.decay:add(1)

      -- averaging
      state.gbars:cmul(state.decay)
      state.gbars:addcmul(1, state.invtaus, dfdx)
      
      state.vbars:cmul(state.decay)
      state.vbars:addcmul(1, state.invtaus, state.dfdx2)

      state.hbars:cmul(state.decay)
      state.hbars:addcmul(1, state.invtaus, diagh)

      -- update
      state.decay:copy(state.gbars)
      state.decay:cmul(state.gbars)
      state.decay:cdiv(state.vbars)
      state.decay:shrink(eps,0)

      state.clr:copy(state.decay)
      state.clr:cdiv(state.hbars)
      x:addcmul(-1, state.clr, dfdx)
   end

   state.nevals = state.nevals + 1

   -- return x after, f(x) before, optimization
   return x,{fx}
end

