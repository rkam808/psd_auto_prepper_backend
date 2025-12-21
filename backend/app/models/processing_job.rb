class ProcessingJob < ApplicationRecord
  enum status: { pending: 0, processing: 1, completed: 2, failed: 3 }

  has_one_attached :original_image
  has_one_attached :prepped_psd
end
