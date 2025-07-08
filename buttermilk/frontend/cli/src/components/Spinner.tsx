
import React from 'react';
import { Text } from 'ink';
import Spinner from 'ink-spinner';

const CustomSpinner = () => (
  <Text color="green">
    <Spinner type="dots" />
  </Text>
);

export default CustomSpinner;
