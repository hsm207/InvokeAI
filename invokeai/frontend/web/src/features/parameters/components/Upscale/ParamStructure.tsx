import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { structureChanged } from '../../store/upscaleSlice';

const ParamStructure = () => {
  const structure = useAppSelector((s) => s.upscale.structure);
  const initial = 0;
  const sliderMin = -5;
  const sliderMax = 5;
  const numberInputMin = -5;
  const numberInputMax = 5;
  const coarseStep = 1;
  const fineStep = 1;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const marks = useMemo(() => [sliderMin, 0, sliderMax], [sliderMax, sliderMin]);
  const onChange = useCallback(
    (v: number) => {
      dispatch(structureChanged(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>Structure</FormLabel>
      <CompositeSlider
        value={structure}
        defaultValue={initial}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        marks={marks}
      />
      <CompositeNumberInput
        value={structure}
        defaultValue={initial}
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamStructure);