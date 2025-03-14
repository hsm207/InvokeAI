import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { WorkflowListMenuTrigger } from 'features/nodes/components/sidePanel/WorkflowListMenu/WorkflowListMenuTrigger';
import { WorkflowViewEditToggleButton } from 'features/nodes/components/sidePanel/WorkflowViewEditToggleButton';
import { selectWorkflowMode } from 'features/nodes/store/workflowSlice';
import { WorkflowLibraryMenu } from 'features/workflowLibrary/components/WorkflowLibraryMenu/WorkflowLibraryMenu';
import { memo } from 'react';

import SaveWorkflowButton from './SaveWorkflowButton';

export const ActiveWorkflowNameAndActions = memo(() => {
  const mode = useAppSelector(selectWorkflowMode);

  return (
    <Flex w="full" alignItems="center" gap={1} minW={0}>
      <WorkflowListMenuTrigger />
      <Spacer />
      {mode === 'edit' && <SaveWorkflowButton />}
      <WorkflowViewEditToggleButton />
      <WorkflowLibraryMenu />
    </Flex>
  );
});
ActiveWorkflowNameAndActions.displayName = 'ActiveWorkflowNameAndActions';
