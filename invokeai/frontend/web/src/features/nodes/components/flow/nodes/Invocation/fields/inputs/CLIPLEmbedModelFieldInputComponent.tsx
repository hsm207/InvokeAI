import { Combobox, Flex, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldCLIPLEmbedValueChanged } from 'features/nodes/store/nodesSlice';
import type { CLIPLEmbedModelFieldInputInstance, CLIPLEmbedModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useCLIPEmbedModels } from 'services/api/hooks/modelsByType';
import { type CLIPLEmbedModelConfig, isCLIPLEmbedModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<CLIPLEmbedModelFieldInputInstance, CLIPLEmbedModelFieldInputTemplate>;

const CLIPLEmbedModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const { t } = useTranslation();
  const disabledTabs = useAppSelector((s) => s.config.disabledTabs);
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useCLIPEmbedModels();

  const _onChange = useCallback(
    (value: CLIPLEmbedModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldCLIPLEmbedValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );
  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs: modelConfigs.filter((config) => isCLIPLEmbedModelConfig(config)),
    onChange: _onChange,
    isLoading,
    selectedModel: field.value,
  });
  const required = props.fieldTemplate.required;

  return (
    <Flex w="full" alignItems="center" gap={2}>
      <Tooltip label={!disabledTabs.includes('models') && t('modelManager.starterModelsInModelManager')}>
        <FormControl className="nowheel nodrag" isDisabled={!options.length} isInvalid={!value && required}>
          <Combobox
            value={value}
            placeholder={required ? placeholder : `(Optional) ${placeholder}`}
            options={options}
            onChange={onChange}
            noOptionsMessage={noOptionsMessage}
          />
        </FormControl>
      </Tooltip>
    </Flex>
  );
};

export default memo(CLIPLEmbedModelFieldInputComponent);