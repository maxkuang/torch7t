-- Stochastic Meta-Descent
-- Author: Sixin Zhang (zsx@cims.nyu.edu)
-- Thanks to Nicol N. Schraudolph, Durk Kingma

----------------------------------------------------------------------
require 'optim'

function optim.smd(opfunc, w, state)

   local state = state or {}
   state.lr = state.lr or 1e-3 -- learning rate
   state.mr = state.mr or 1e-2 -- meta rate
   state.td = state.td or 0.99 -- td lambda
   state.f = state.f or inf              -- loss
   state.g = state.g or w:clone():zero() -- grad, temp
   state.v = state.v or w:clone():zero() -- dwdp
   state.hv = state.hv or w:clone():zero() -- Hv, temp

   -- evaluate
   local f,g = opfunc(w)

   state.f = f
   state.g:copy(g)

   -- Hv
   local r = 1e-3
   state.hv:copy(w)
   state.hv:add(r,state.v) -- w+r*v
   opfunc(state.hv) -- g -> g@(w+r*v), f -> f@(w+r*v)

   state.hv:copy(g) -- g@(w+r*v)
   state.hv:add(-1,state.g)
   state.hv:div(r) -- Hv

   -- v
   state.hv:mul(state.td) -- td*
   state.hv:add(1,g) -- g+
   state.hv:mul(-state.lr) -- lr-
   state.hv:add(state.td,state.v) -- v = td*v-lr*(g+td*Hv)

   -- w
   w:add(-state.lr,state.g)

   -- lr
   state.lr = state.lr * math.exp(-state.mr*(state.g:cmul(state.v):sum()))

   -- v
   state.v:copy(state.hv)

   return w,{state.f}
end

-- smd
print '==> smd'
optim_method = optim.smd
optim_state = {
   lr = opt.lr,
   mr = opt.mr,
   td = opt.td,
}
optim_logger = function()
   logger:add{['lr'] = optim_state.lr}
end
