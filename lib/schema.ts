import { z } from "zod";

export const DamageItem = z.object({
  area: z.enum(["front","front-left","left","rear-left","rear","rear-right","right","front-right","roof","unknown"]),
  part: z.enum(["bumper","fender","door","hood","trunk","quarter-panel","headlight","taillight","grille","mirror","windshield","wheel","unknown"]),
  type: z.enum(["dent","scratch","crack","paint-chips","broken","bent","missing","glass-crack","unknown"]),
  severity: z.number().min(1).max(5),
  confidence: z.number().min(0).max(1),
  bbox: z.array(z.number()).length(4).optional()
});

export const AnalysisSchema = z.object({
  vehicle: z.object({
    make: z.string().nullable(),
    model: z.string().nullable(),
    color: z.string().nullable(),
    confidence: z.number()
  }),
  damage: z.array(DamageItem).default([]),
  quality: z.object({
    isVehicle: z.boolean(),
    isCar: z.boolean(),
    conf: z.number()
  })
});
