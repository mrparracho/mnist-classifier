import React from 'react';
import { Box, Text, VStack, HStack, Badge } from '@chakra-ui/react';

interface FeedbackDisplayProps {
  isCorrect: boolean | null;
  predictedDigit: number | null;
  actualDigit: number | null;
}

const FeedbackDisplay: React.FC<FeedbackDisplayProps> = ({
  isCorrect,
  predictedDigit,
  actualDigit,
}) => {
  if (isCorrect === null) return null;

  const feedbackColor = isCorrect ? 'green.500' : 'red.500';
  const feedbackText = isCorrect ? 'Correct!' : 'Incorrect';

  return (
    <Box
      p={4}
      borderRadius="lg"
      bg={isCorrect ? 'green.50' : 'red.50'}
      border="1px solid"
      borderColor={feedbackColor}
      width="100%"
      maxW="400px"
      my={4}
    >
      <VStack spacing={3} align="stretch">
        <HStack justify="space-between">
          <Text fontWeight="bold" color={feedbackColor}>
            {feedbackText}
          </Text>
          <Badge colorScheme={isCorrect ? 'green' : 'red'} fontSize="0.8em">
            {isCorrect ? '✓' : '✗'}
          </Badge>
        </HStack>
        
        <HStack justify="space-between" spacing={4}>
          <Box>
            <Text fontSize="sm" color="gray.600">
              Predicted:
            </Text>
            <Text fontSize="lg" fontWeight="bold">
              {predictedDigit}
            </Text>
          </Box>
          
          <Box>
            <Text fontSize="sm" color="gray.600">
              Actual:
            </Text>
            <Text fontSize="lg" fontWeight="bold">
              {actualDigit}
            </Text>
          </Box>
        </HStack>
      </VStack>
    </Box>
  );
};

export default FeedbackDisplay; 